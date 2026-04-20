"""
QR Code Detection and Decoding System - (mostly) From Scratch
=====================================================
Detect and decode QR codes, without using explicit QR libraries.
Allowed libs: opencv-python, numpy
"""

import argparse
import csv
import itertools
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Worker globals (set once per worker process via initializer) ───────────────
_worker_detector = None
_worker_decoder = None

def _worker_init(no_decode: bool):
    """Initializer for ProcessPoolExecutor workers — creates one detector/decoder
    per process so we pay construction cost only once, not per image.
    cv2.setNumThreads(1) prevents each worker from spawning its own OpenCV thread
    pool, which would cause oversubscription and 4× slowdown with 4 workers."""
    global _worker_detector, _worker_decoder
    cv2.setNumThreads(1)
    _worker_detector = QRDetector()
    _worker_decoder = None if no_decode else QRDecoder()

def _process_image_worker(args):
    """Top-level (picklable) worker function called by ProcessPoolExecutor."""
    path, image_id = args
    return process_image(path, image_id, _worker_detector, _worker_decoder, verbose=False)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: GALOIS FIELD GF(256) ARITHMETIC
# ══════════════════════════════════════════════════════════════════════════════

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256

def _init_gf():
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]

_init_gf()

def gf_mul(a, b):
    return 0 if (a == 0 or b == 0) else _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]

def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError
    return 0 if a == 0 else _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]

def gf_pow(x, p):
    return 0 if x == 0 else _GF_EXP[(_GF_LOG[x] * p) % 255]

def gf_inv(x):
    if x == 0:
        raise ZeroDivisionError
    return _GF_EXP[255 - _GF_LOG[x]]

def gf_poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            r[i + j] ^= gf_mul(pi, qj)
    return r

def gf_poly_eval(poly, x):
    r = 0
    for c in poly:
        r = gf_mul(r, x) ^ c
    return r

def berlekamp_massey(s):
    n = len(s)
    C = [1]
    B = [1]
    L, m, b = 0, 1, 1
    for i in range(n):
        d = s[i]
        for j in range(1, L + 1):
            if j < len(C):
                d ^= gf_mul(C[j], s[i - j])
        if d == 0:
            m += 1
        elif 2 * L <= i:
            T = C[:]
            cf = gf_div(d, b)
            while len(C) < len(B) + m:
                C.append(0)
            for j, bj in enumerate(B):
                C[j + m] ^= gf_mul(cf, bj)
            L = i + 1 - L
            B, b, m = T, d, 1
        else:
            cf = gf_div(d, b)
            while len(C) < len(B) + m:
                C.append(0)
            for j, bj in enumerate(B):
                C[j + m] ^= gf_mul(cf, bj)
            m += 1
    return C

def rs_decode(received, n_ec):
    n = len(received)
    k = n - n_ec
    syn = [gf_poly_eval(received, gf_pow(2, i)) for i in range(n_ec)]
    if all(s == 0 for s in syn):
        return list(received[:k])
    err_loc = berlekamp_massey(syn)
    if len(err_loc) - 1 > n_ec // 2:
        return None
    err_pos = [n - 1 - i for i in range(n) if gf_poly_eval(err_loc, gf_pow(2, i)) == 0]
    if len(err_pos) != len(err_loc) - 1:
        return None
    result = list(received)
    # Forney algorithm
    omega = gf_poly_mul(syn[::-1], err_loc)[:n_ec][::-1]
    for pos in err_pos:
        if not (0 <= pos < n):
            return None
        xi = gf_pow(2, pos)
        xi_inv = gf_inv(xi)
        ov = gf_poly_eval(omega, xi_inv)
        # formal derivative of error locator evaluated at xi_inv
        sp = 0
        for idx in range(0, len(err_loc), 2):
            sp ^= gf_mul(err_loc[idx], gf_pow(xi_inv, idx))
        if sp == 0:
            return None
        result[pos] ^= gf_div(gf_mul(xi, ov), sp)
    if all(gf_poly_eval(result, gf_pow(2, i)) == 0 for i in range(n_ec)):
        return result[:k]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: QR CODE STANDARD TABLES
# ══════════════════════════════════════════════════════════════════════════════

_ALPHA_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:' 

_MASK_FN = [
    lambda r, c: (r + c) % 2 == 0,
    lambda r, c: r % 2 == 0,
    lambda r, c: c % 3 == 0,
    lambda r, c: (r + c) % 3 == 0,
    lambda r, c: (r // 2 + c // 3) % 2 == 0,
    lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
    lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
    lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0,
]

_AP_INTERVALS = [
    [],  # version 1
    [6, 18], [6, 22], [6, 26], [6, 30], [6, 34],
    [6, 22, 38], [6, 24, 42], [6, 26, 46], [6, 28, 50],
    [6, 30, 54], [6, 32, 58], [6, 34, 62], [6, 26, 46, 66],
    [6, 26, 48, 70], [6, 26, 50, 74], [6, 30, 54, 78],
    [6, 30, 56, 82], [6, 30, 58, 86], [6, 34, 62, 90],
    [6, 28, 50, 72, 94], [6, 26, 50, 74, 98],
    [6, 30, 54, 78, 102], [6, 28, 54, 80, 106],
    [6, 32, 58, 84, 110], [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118], [6, 26, 50, 74, 98, 122],
    [6, 30, 54, 78, 102, 126], [6, 26, 52, 78, 104, 130],
    [6, 30, 56, 82, 108, 134], [6, 34, 60, 86, 112, 138],
    [6, 30, 58, 86, 114, 142], [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150], [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158], [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166], [6, 30, 58, 86, 114, 142, 170],
]

_FORMAT_MASK = 0x5412

# ECC table: version -> ec_level -> list of (num_blocks, data_cw_per_block, ec_cw_per_block)
# Format: {version: {'L': [(blocks, data_cw, ec_cw), ...], 'M': [...], 'Q': [...], 'H': [...]}}
_ECC_TABLE = {
    1:  {'L': [(1, 19, 7)],    'M': [(1, 16, 10)],   'Q': [(1, 13, 13)],   'H': [(1, 9, 17)]},
    2:  {'L': [(1, 34, 10)],   'M': [(1, 28, 16)],   'Q': [(1, 22, 22)],   'H': [(1, 16, 28)]},
    3:  {'L': [(1, 55, 15)],   'M': [(1, 44, 26)],   'Q': [(2, 17, 18)],   'H': [(2, 13, 22)]},
    4:  {'L': [(1, 80, 20)],   'M': [(2, 32, 18)],   'Q': [(2, 24, 26)],   'H': [(4, 9, 16)]},
    5:  {'L': [(1, 108, 26)],  'M': [(2, 43, 24)],   'Q': [(2, 15, 18), (2, 16, 18)], 'H': [(2, 11, 22), (2, 12, 22)]},
    6:  {'L': [(2, 68, 18)],   'M': [(4, 27, 16)],   'Q': [(4, 19, 24)],   'H': [(4, 15, 28)]},
    7:  {'L': [(2, 78, 20)],   'M': [(4, 31, 18)],   'Q': [(2, 14, 18), (4, 15, 18)], 'H': [(4, 13, 26), (1, 14, 26)]},
    8:  {'L': [(2, 97, 24)],   'M': [(2, 38, 22), (2, 39, 22)], 'Q': [(4, 18, 22), (2, 19, 22)], 'H': [(4, 14, 26), (2, 15, 26)]},
    9:  {'L': [(2, 116, 30)],  'M': [(3, 36, 22), (2, 37, 22)], 'Q': [(4, 16, 20), (4, 17, 20)], 'H': [(4, 12, 24), (4, 13, 24)]},
    10: {'L': [(2, 68, 18), (2, 69, 18)], 'M': [(4, 43, 26), (1, 44, 26)], 'Q': [(6, 19, 24), (2, 20, 24)], 'H': [(6, 15, 28), (2, 16, 28)]},
    11: {'L': [(4, 81, 20)],   'M': [(1, 50, 30), (4, 51, 30)], 'Q': [(4, 22, 28), (4, 23, 28)], 'H': [(3, 12, 24), (8, 13, 24)]},
    12: {'L': [(2, 92, 24), (2, 93, 24)], 'M': [(6, 36, 22), (2, 37, 22)], 'Q': [(4, 20, 26), (6, 21, 26)], 'H': [(7, 14, 28), (4, 15, 28)]},
    13: {'L': [(4, 107, 26)],  'M': [(8, 37, 22), (1, 38, 22)], 'Q': [(8, 20, 24), (4, 21, 24)], 'H': [(12, 11, 22), (4, 12, 22)]},
    14: {'L': [(3, 115, 30), (1, 116, 30)], 'M': [(4, 40, 24), (5, 41, 24)], 'Q': [(11, 16, 20), (5, 17, 20)], 'H': [(11, 12, 24), (5, 13, 24)]},
    15: {'L': [(5, 87, 22), (1, 88, 22)], 'M': [(5, 41, 24), (5, 42, 24)], 'Q': [(5, 24, 30), (7, 25, 30)], 'H': [(11, 12, 24), (7, 13, 24)]},
    16: {'L': [(5, 98, 24), (1, 99, 24)], 'M': [(7, 45, 28), (3, 46, 28)], 'Q': [(15, 19, 24), (2, 20, 24)], 'H': [(3, 15, 30), (13, 16, 30)]},
    17: {'L': [(1, 107, 28), (5, 108, 28)], 'M': [(10, 46, 28), (1, 47, 28)], 'Q': [(1, 22, 28), (15, 23, 28)], 'H': [(2, 14, 28), (17, 15, 28)]},
    18: {'L': [(5, 120, 30), (1, 121, 30)], 'M': [(9, 43, 26), (4, 44, 26)], 'Q': [(17, 22, 28), (1, 23, 28)], 'H': [(2, 14, 28), (19, 15, 28)]},
    19: {'L': [(3, 113, 28), (4, 114, 28)], 'M': [(3, 44, 26), (11, 45, 26)], 'Q': [(17, 21, 26), (4, 22, 26)], 'H': [(9, 13, 26), (16, 14, 26)]},
    20: {'L': [(3, 107, 28), (5, 108, 28)], 'M': [(3, 41, 26), (13, 42, 26)], 'Q': [(15, 24, 30), (5, 25, 30)], 'H': [(15, 15, 28), (10, 16, 28)]},
    21: {'L': [(4, 116, 28), (4, 117, 28)], 'M': [(17, 42, 26)],             'Q': [(17, 22, 28), (6, 23, 28)], 'H': [(19, 16, 30), (6, 17, 30)]},
    22: {'L': [(2, 111, 28), (7, 112, 28)], 'M': [(17, 46, 28)],             'Q': [(7, 24, 30), (16, 25, 30)], 'H': [(34, 13, 24)]},
    23: {'L': [(4, 121, 30), (5, 122, 30)], 'M': [(4, 47, 28), (14, 48, 28)], 'Q': [(11, 24, 30), (14, 25, 30)], 'H': [(16, 15, 30), (14, 16, 30)]},
    24: {'L': [(6, 117, 30), (4, 118, 30)], 'M': [(6, 45, 28), (14, 46, 28)], 'Q': [(11, 24, 30), (16, 25, 30)], 'H': [(30, 16, 30), (2, 17, 30)]},
    25: {'L': [(8, 106, 26), (4, 107, 26)], 'M': [(8, 47, 28), (13, 48, 28)], 'Q': [(7, 24, 30), (22, 25, 30)], 'H': [(22, 15, 30), (13, 16, 30)]},
    26: {'L': [(10, 114, 28), (2, 115, 28)], 'M': [(19, 46, 28), (4, 47, 28)], 'Q': [(28, 22, 28), (6, 23, 28)], 'H': [(33, 16, 30), (4, 17, 30)]},
    27: {'L': [(8, 122, 30), (4, 123, 30)], 'M': [(22, 45, 28), (3, 46, 28)], 'Q': [(8, 23, 28), (26, 24, 28)], 'H': [(12, 15, 30), (28, 16, 30)]},
    28: {'L': [(3, 117, 30), (10, 118, 30)], 'M': [(3, 45, 28), (23, 46, 28)], 'Q': [(4, 24, 30), (31, 25, 30)], 'H': [(11, 15, 30), (31, 16, 30)]},
    29: {'L': [(7, 116, 30), (7, 117, 30)], 'M': [(21, 45, 28), (7, 46, 28)], 'Q': [(1, 23, 28), (37, 24, 28)], 'H': [(19, 15, 30), (26, 16, 30)]},
    30: {'L': [(5, 115, 30), (10, 116, 30)], 'M': [(19, 47, 28), (10, 48, 28)], 'Q': [(15, 24, 30), (25, 25, 30)], 'H': [(23, 15, 30), (25, 16, 30)]},
    31: {'L': [(13, 115, 30), (3, 116, 30)], 'M': [(2, 46, 28), (29, 47, 28)], 'Q': [(42, 24, 30), (1, 25, 30)], 'H': [(23, 15, 30), (28, 16, 30)]},
    32: {'L': [(17, 115, 30)], 'M': [(10, 46, 28), (23, 47, 28)], 'Q': [(10, 24, 30), (35, 25, 30)], 'H': [(19, 15, 30), (35, 16, 30)]},
    33: {'L': [(17, 115, 30), (1, 116, 30)], 'M': [(14, 46, 28), (21, 47, 28)], 'Q': [(29, 24, 30), (19, 25, 30)], 'H': [(11, 15, 30), (46, 16, 30)]},
    34: {'L': [(13, 115, 30), (6, 116, 30)], 'M': [(14, 46, 28), (23, 47, 28)], 'Q': [(44, 24, 30), (7, 25, 30)], 'H': [(59, 16, 30), (1, 17, 30)]},
    35: {'L': [(12, 121, 30), (7, 122, 30)], 'M': [(12, 47, 28), (26, 48, 28)], 'Q': [(39, 24, 30), (14, 25, 30)], 'H': [(22, 15, 30), (41, 16, 30)]},
    36: {'L': [(6, 121, 30), (14, 122, 30)], 'M': [(6, 47, 28), (34, 48, 28)], 'Q': [(46, 24, 30), (10, 25, 30)], 'H': [(2, 15, 30), (64, 16, 30)]},
    37: {'L': [(17, 122, 30), (4, 123, 30)], 'M': [(29, 46, 28), (14, 47, 28)], 'Q': [(49, 24, 30), (10, 25, 30)], 'H': [(24, 15, 30), (46, 16, 30)]},
    38: {'L': [(4, 122, 30), (18, 123, 30)], 'M': [(13, 46, 28), (32, 47, 28)], 'Q': [(48, 24, 30), (14, 25, 30)], 'H': [(42, 15, 30), (32, 16, 30)]},
    39: {'L': [(20, 117, 30), (4, 118, 30)], 'M': [(40, 47, 28), (7, 48, 28)], 'Q': [(43, 24, 30), (22, 25, 30)], 'H': [(10, 15, 30), (67, 16, 30)]},
    40: {'L': [(19, 118, 30), (6, 119, 30)], 'M': [(18, 47, 28), (31, 48, 28)], 'Q': [(34, 24, 30), (34, 25, 30)], 'H': [(20, 15, 30), (61, 16, 30)]},
}

_EC_LEVEL_INV = {1: 'L', 0: 'M', 3: 'Q', 2: 'H'}

_CCB = {
    'N': {(1, 9): 10, (10, 26): 12, (27, 40): 14},
    'A': {(1, 9): 9, (10, 26): 11, (27, 40): 13},
    'B': {(1, 9): 8, (10, 26): 16, (27, 40): 16},
    'K': {(1, 9): 8, (10, 26): 10, (27, 40): 12},
}

def _ccb(mode, ver):
    for (lo, hi), b in _CCB.get(mode, {}).items():
        if lo <= ver <= hi:
            return b
    return 8


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: QR CODE DETECTION (FINDER PATTERN BASED)
# ══════════════════════════════════════════════════════════════════════════════

_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # pre-built kernel

def _binaries_lazy(gray):
    """Yield binary images one at a time, ordered cheapest-first.
    This allows early-exit when a valid QR is found with cheap strategies.
    Cost order: Otsu (5ms) < Mean-21 (12ms) < block-25 (14ms) < block-15 (24ms)
                < block-7 (34ms) < block-11 (42ms)  [measured at 640x640px].
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    geq = clahe.apply(gray)
    blur_g = cv2.GaussianBlur(geq, (3, 3), 0)

    # Cheapest: Otsu (global threshold, few contours)
    _, b = cv2.threshold(blur_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    yield b

    # Mean adaptive block=21 (larger blocks → fewer contours)
    yield cv2.adaptiveThreshold(blur_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 4)

    # block=25 + morph-close
    b5r = cv2.adaptiveThreshold(blur_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 2)
    yield cv2.morphologyEx(b5r, cv2.MORPH_CLOSE, _K3)

    # Sharpened + block=15
    blur_s = cv2.GaussianBlur(geq, (0, 0), 3)
    geq_sharp = np.clip(cv2.addWeighted(geq, 1.8, blur_s, -0.8, 0), 0, 255).astype(np.uint8)
    yield cv2.adaptiveThreshold(cv2.GaussianBlur(geq_sharp, (3, 3), 0), 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)

    # Raw + block=7 (expensive — fine detail, many tiny contours)
    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 7, 2)

    # CLAHE + block=11 (most expensive but most common use-case)
    yield cv2.adaptiveThreshold(geq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)




def _find_finder_candidates(binary):
    """Find finder pattern candidates using 3-level contour hierarchy nesting.
    Optimized: early-reject via bounding rect (cheap) before contourArea (slow).
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return []
    hier = hierarchy[0]
    cands = []
    # Prerequisite: has a child (no children = leaf, can't be finder outer)
    # This early filter saves 30-40% of contourArea calls since many contours
    # are leaves (e.g., noise, text pixels).
    for i, cnt in enumerate(contours):
        ci = hier[i][2]
        if ci < 0:
            continue  # no child
        # Bounding-rect gate first — much cheaper than contourArea.
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 3 or bh < 3:
            continue
        if max(bw, bh) / (min(bw, bh) + 1e-6) > 4.5:
            continue
        # Now contourArea (still cheap compared to findContours):
        area = cv2.contourArea(cnt)
        if area < 9:
            continue
        # Grandchild must also exist for 3-level nesting
        gi = hier[ci][2]
        if gi < 0:
            continue
        ca = cv2.contourArea(contours[ci])
        ratio_om = area / (ca + 1e-6)
        if ca < 2 or ratio_om > 12 or ratio_om < 1.2:
            continue
        ga = cv2.contourArea(contours[gi])
        ratio_mi = ca / (ga + 1e-6)
        if ga < 1 or ratio_mi > 12 or ratio_mi < 1.2:
            continue

        cx = x + bw // 2
        cy = y + bh // 2
        side = max(bw, bh)
        cands.append({'center': (cx, cy), 'size': area,
                      'bw': bw, 'bh': bh, 'side': float(side)})
    return cands


def _check_ratio_runs(runs, tol=0.85):
    """Check if runs match 1:1:3:1:1 ratio of finder pattern."""
    if len(runs) < 5:
        return False
    total = sum(runs)
    if total == 0:
        return False
    u = total / 7.0
    return all(abs(r - e * u) <= (tol * u + 1.0) for r, e in zip(runs, [1, 1, 3, 1, 1]))


def _scan_line_verify_np(binary, cx, cy, radius):
    """Vectorized scan-line 1:1:3:1:1 verification using numpy.
    Returns number of directions (out of 4) that pass the ratio test.
    """
    h, w = binary.shape
    passed = 0
    ts = np.arange(-radius, radius + 1)

    for dx, dy in [(1, 0), (0, 1), (0.707, 0.707), (0.707, -0.707)]:
        rs = np.clip((cy + ts * dy).astype(int), 0, h - 1)
        cs = np.clip((cx + ts * dx).astype(int), 0, w - 1)
        line = (binary[rs, cs] > 127).astype(np.int8)
        # Run-length encode
        changes = np.where(np.diff(line) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(line)]])
        runs = (ends - starts).tolist()
        if any(_check_ratio_runs(runs[j:j+5]) for j in range(len(runs) - 4)):
            passed += 1
    return passed


def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))


def _angle_at(vertex, p1, p2):
    v1 = np.array(p1, float) - np.array(vertex, float)
    v2 = np.array(p2, float) - np.array(vertex, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def _order_corners(pts):
    """Return corners in CW order (TL, TR, BR, BL). For already-ordered input
    this is the identity; otherwise we sort by polar angle around the centroid
    which stays correct under arbitrary rotation. The simple argmin-of-sum
    scheme breaks for QRs rotated ~45° (two corners collide onto the same
    extremum and the quadrilateral degenerates to a line).
    """
    pts = np.array(pts, dtype=np.float32)
    if len(pts) != 4:
        return pts
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    # angle from centroid; sort CCW (math convention), then pick TL as the
    # point closest to the image-origin direction.
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    ordered = pts[order]
    # Rotate so that the "top-left-most" (smallest x+y) point is first.
    sums = ordered.sum(axis=1)
    start = int(np.argmin(sums))
    ordered = np.roll(ordered, -start, axis=0)
    # Our CCW sort yields [TL, BL, BR, TR]; swap to produce CW [TL, TR, BR, BL].
    return np.array([ordered[0], ordered[3], ordered[2], ordered[1]],
                    dtype=np.float32)


def _point_in_quad(point, corners):
    """Return True if point is strictly inside the quadrilateral (corners ordered CW or CCW)."""
    px, py = float(point[0]), float(point[1])
    n = len(corners)
    pos = neg = 0
    for idx in range(n):
        x1, y1 = float(corners[idx][0]), float(corners[idx][1])
        x2, y2 = float(corners[(idx + 1) % n][0]), float(corners[(idx + 1) % n][1])
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 0:
            pos += 1
        elif cross < 0:
            neg += 1
    return pos == n or neg == n


def _count_points_in_quad(points_np, corners):
    """Vectorised: count how many points (N×2 numpy array) lie strictly
    inside the quadrilateral `corners` (4×2). Returns an int."""
    if points_np.size == 0:
        return 0
    cs = np.asarray(corners, dtype=np.float64)
    pos = np.zeros(points_np.shape[0], dtype=np.int64)
    neg = np.zeros(points_np.shape[0], dtype=np.int64)
    for k in range(4):
        x1, y1 = cs[k]
        x2, y2 = cs[(k + 1) % 4]
        cross = (x2 - x1) * (points_np[:, 1] - y1) - (y2 - y1) * (points_np[:, 0] - x1)
        pos += (cross > 0).astype(np.int64)
        neg += (cross < 0).astype(np.int64)
    return int(np.sum((pos == 4) | (neg == 4)))


def _is_duplicate_qr(qr1_corners, qr2_corners):
    c1 = np.mean(qr1_corners, axis=0)
    c2 = np.mean(qr2_corners, axis=0)
    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    s1 = np.linalg.norm(np.array(qr1_corners[0]) - np.array(qr1_corners[1]))
    s2 = np.linalg.norm(np.array(qr2_corners[0]) - np.array(qr2_corners[1]))
    side_len = max(s1, s2)
    if side_len < 1:
        return True
    return dist < (side_len * 0.4)


def _validate_corners(corners):
    if corners is None:
        return False
    pts = np.array(corners, dtype=np.float32)
    sides = [_dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 6:
        return False
    if max(sides) / (min(sides) + 1e-6) > 5.0:
        return False
    d1 = _dist(pts[0], pts[2])
    d2 = _dist(pts[1], pts[3])
    if max(d1, d2) / (min(d1, d2) + 1e-6) > 5.0:
        return False
    return True


def _best_module_size(dist_h, dist_v, avg_finder_side):
    """Pick the module size (in px) that best reconciles the finder-centre
    distance with a plausible QR version. `avg_finder_side` should be the
    bounding-box side (not sqrt of area — contour area under-measures the
    7×7 outer ring by 30–50 % because `cv2.contourArea` of the outer
    contour only traces its 4-connected outline on the inverted binary).
    """
    dist_avg = (dist_h + dist_v) / 2.0
    best_mod = avg_finder_side / 7.0
    best_err = float('inf')
    for v in range(1, 41):
        n = 4 * v + 10  # centre-to-centre in modules
        m_d = dist_avg / n
        # Reference module from finder bounding box (7 modules across)
        m_f = avg_finder_side / 7.0
        err = abs(m_d - m_f)
        if err < best_err:
            best_err = err
            best_mod = m_d
    return best_mod


def _estimate_corners(tl_pt, a_pt, b_pt, avg_finder_side):
    tl = np.array(tl_pt, dtype=float)
    a = np.array(a_pt, dtype=float)
    b = np.array(b_pt, dtype=float)
    va = a - tl
    da = np.linalg.norm(va)
    vb = b - tl
    db = np.linalg.norm(vb)
    if da < 1 or db < 1:
        return None

    # Determine which is TR and which is BL
    va_cw90 = np.array([va[1], -va[0]])
    if np.dot(va_cw90, vb) > 0:
        tr, bl = a, b
        rv, dv = va, vb
    else:
        tr, bl = b, a
        rv, dv = vb, va

    dist_h = float(np.linalg.norm(rv))
    dist_v = float(np.linalg.norm(dv))
    if dist_h < 1 or dist_v < 1:
        return None

    ru = rv / dist_h
    du = dv / dist_v
    mod = _best_module_size(dist_h, dist_v, avg_finder_side)
    # 3.5 modules = half a finder pattern (centre → outer QR edge).
    # Many GT annotations include a small margin beyond the QR data area
    # (up to the first module of the quiet zone). Adding ~1 extra module
    # on each side gives better IoU overlap without destroying precision
    # on tightly-labelled images.
    offset = 4.0 * mod

    c_tl = tl - offset * ru - offset * du
    c_tr = tr + offset * ru - offset * du
    c_bl = bl - offset * ru + offset * du
    c_br_para = c_tr + (c_bl - c_tl)
    c_br_tr = c_tr + (dist_v + 2 * offset) * du
    c_br_bl = c_bl + (dist_h + 2 * offset) * ru
    c_br_ext = (c_br_tr + c_br_bl) / 2.0
    c_br = 0.6 * c_br_para + 0.4 * c_br_ext

    return _order_corners([c_tl, c_tr, c_br, c_bl])


def _merge_candidates(cands_list, merge_factor=0.6):
    """Merge duplicate candidates across strategies.
    merge_factor controls the merge radius (× sqrt(finder area)).
    Use a smaller factor (e.g. 0.5–0.7) for dense grids where neighbouring
    QR finders are close; larger factor (1.0+) only for sparse scenes.

    Vectorised: O(n²) work is pushed into numpy so large candidate sets
    (100+) merge in tens of ms instead of seconds.
    """
    all_cands = [c for cands in cands_list for c in cands]
    n = len(all_cands)
    if n == 0:
        return []
    if n == 1:
        return list(all_cands)
    centers = np.array([c['center'] for c in all_cands], dtype=np.float32)
    sizes = np.array([c['size'] for c in all_cands], dtype=np.float32)
    sqsz = np.sqrt(sizes)
    # threshold matrix: thr[i,j] = max(sqsz[i], sqsz[j]) * merge_factor
    thr = np.maximum(sqsz[:, None], sqsz[None, :]) * merge_factor
    # squared distance matrix
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    mask = d2 < (thr * thr)
    # Greedy merge: sort by -size, pick seeds, absorb neighbours.
    order = np.argsort(-sizes)
    used = np.zeros(n, dtype=bool)
    merged = []
    for i in order:
        if used[i]:
            continue
        # candidate i is new cluster seed; mark self + neighbours as used
        nbr = np.where(mask[i] & ~used)[0]
        used[nbr] = True
        merged.append(all_cands[int(i)])
    return merged


def _spatial_cluster(candidates, k_factor=10.0):
    n = len(candidates)
    if n <= 3:
        return [candidates]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sides = [np.sqrt(c['size']) for c in candidates]
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(candidates[i]['center'], candidates[j]['center'])
            if d <= k_factor * max(sides[i], sides[j]):
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(candidates[i])
    return list(clusters.values())


def _check_triplet(candidates, i, j, k, centers, sizes, gray_img, sides=None):
    """Validate a triplet of finder patterns as a potential QR code.
    `sides[]` (optional) is the per-candidate bounding-box side length; when
    provided it is used as a more robust finder scale than sqrt(area).
    """
    pi, pj, pk = centers[i], centers[j], centers[k]
    si, sj, sk = sizes[i], sizes[j], sizes[k]
    avg_size = (si + sj + sk) / 3.0
    # Two estimators for the finder outer side:
    #  - sqrt(area) ≈ 7·m for an axis-aligned finder (outer contour area =
    #    49·m² when finder is fully filled on the inversion). Reliable scale.
    #  - bounding-rect side = 7·m·(cosθ+sinθ) for a rotated finder, so it
    #    overshoots by up to √2. Still useful as a max plausible extent.
    # For distance / timing checks we use the tighter area-based estimate so
    # rotated finders don't get falsely rejected for being "too close".
    avg_side_area = float(np.sqrt(avg_size))
    if sides is not None:
        avg_side_bbox = (sides[i] + sides[j] + sides[k]) / 3.0
    else:
        avg_side_bbox = avg_side_area
    avg_side = avg_side_area  # used for thresholds below

    def size_ok(s1, s2, tol=1.3):
        return max(np.sqrt(s1), np.sqrt(s2)) / (min(np.sqrt(s1), np.sqrt(s2)) + 1e-6) < (1.0 + tol)

    if not (size_ok(si, sj) and size_ok(sj, sk) and size_ok(si, sk)):
        return None

    ai = _angle_at(pi, pj, pk)
    aj = _angle_at(pj, pi, pk)
    ak = _angle_at(pk, pi, pj)
    angles = sorted([(ai, i, pi), (aj, j, pj), (ak, k, pk)], reverse=True)

    if not (40 < angles[0][0] < 140):
        return None
    if angles[2][0] < 8:
        return None

    corner_pt = angles[0][2]
    op = [angles[1][2], angles[2][2]]
    d1 = _dist(corner_pt, op[0])
    d2 = _dist(corner_pt, op[1])
    if d1 < 1 or d2 < 1:
        return None

    side_ratio = max(d1, d2) / min(d1, d2)
    if side_ratio > 3.0:
        return None

    d_hyp = _dist(op[0], op[1])
    hyp_ratio = d_hyp / ((d1 + d2) / 2.0 + 1e-6)
    if not (0.85 < hyp_ratio < 2.2):
        return None

    # Distance lower-bound: V1 QR has centre-to-centre 14 modules = 2×
    # finder_side (finder = 7 modules). Require d >= 1.6× finder_side; the
    # extra 0.1 margin handles noise / perspective distortion.
    finder_side = avg_side
    if d1 < finder_side * 1.6 or d2 < finder_side * 1.6:
        return None
    # V40 QR: centre-to-centre = (4·40+17-7) = 170 modules = 24.3× finder_side.
    # In practice the `avg_side_area = sqrt(contour_area)` underestimates
    # finder size for high-version codes (tiny module, antialiased contour
    # area understates the 7·m outer extent). Cap generously at 34× so V40
    # codes with sloppy area measurements still pass.
    if d1 > finder_side * 34.0 or d2 > finder_side * 34.0:
        return None

    _mod = finder_side / 7.0
    if _mod < 0.1:
        return None

    # Timing pattern check — samples the line between two finder centres
    # (skipping the 3.5-module-wide finder patterns on each end), thresholds
    # locally, and counts dark↔light transitions. A valid QR timing pattern
    # alternates every module, giving transitions ≈ d/mod − 8. Bogus triplets
    # whose "timing line" passes through QR data or white space will have
    # either too few or too many transitions.
    def check_timing(p1, p2, d):
        if d < 7 * _mod:
            return True
        u = (np.array(p2) - np.array(p1)) / d
        start = np.array(p1) + u * (4.0 * _mod)
        end = np.array(p2) - u * (4.0 * _mod)
        length = int(np.hypot(end[0] - start[0], end[1] - start[1]))
        if length < 3:
            return True
        x = np.linspace(start[0], end[0], length)
        y = np.linspace(start[1], end[1], length)
        pixels = gray_img[np.clip(y.astype(int), 0, gray_img.shape[0] - 1),
                          np.clip(x.astype(int), 0, gray_img.shape[1] - 1)]
        if len(pixels) == 0:
            return True
        v_min, v_max = int(np.min(pixels)), int(np.max(pixels))
        if v_max - v_min < 8:
            return False
        thresh = (v_max + v_min) / 2
        binary = (pixels > thresh).astype(np.uint8)
        transitions = int(np.sum(binary[:-1] != binary[1:]))
        expected = max(1.0, d / _mod - 8)
        # Accept a wide but bounded range scaled by expected count:
        # - For short lines (V1 QRs, only ~4 data modules between finders)
        #   a real QR can legitimately show 0-2 transitions depending on
        #   the mask bits, so we relax the lower bound to 0.
        # - For long lines (V3+) we require ≥0.25×expected to reject
        #   fakes whose line passes through mostly-white gaps.
        # - The upper bound (≤1.8×expected) catches fakes whose line
        #   crosses two adjacent QR codes in a dense grid.
        if expected < 6:
            return transitions <= max(2, expected * 1.8)
        ratio = transitions / expected
        return 0.20 <= ratio <= 1.8

    if not check_timing(corner_pt, op[0], d1):
        return None
    if not check_timing(corner_pt, op[1], d2):
        return None

    # Version estimate: use BOTH area- and bbox-based module sizes; accept the
    # triplet if either yields a plausible QR version. The area-based _mod
    # tends to undershoot for high-version codes (antialiased tiny modules),
    # while bbox-based overshoots for rotated finders — testing both handles
    # the full range V1..V40.
    _N_h = d1 / _mod + 7
    _N_v = d2 / _mod + 7
    _v_est = ((_N_h + _N_v) / 2.0 - 17) / 4.0
    _mod_bbox = avg_side_bbox / 7.0
    _v_est_b = ((d1 / _mod_bbox + d2 / _mod_bbox) / 2.0 + 7 - 17) / 4.0
    if not (0.3 <= _v_est <= 41.0 or 0.3 <= _v_est_b <= 41.0):
        return None

    v0 = np.array(op[0], float) - np.array(corner_pt, float)
    v1 = np.array(op[1], float) - np.array(corner_pt, float)
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 > 1e-6 and n1 > 1e-6:
        if abs(np.dot(v0, v1) / (n0 * n1)) > 0.80:
            return None

    size_var = np.std([si, sj, sk]) / (avg_size + 1e-6)
    score = (abs(angles[0][0] - 90) * 2.5 + (side_ratio - 1.0) * 100 +
             abs(hyp_ratio - 1.414) * 100 + size_var * 60)
    # Hand the BBOX side down for corner extrapolation (gives larger corners
    # that better match GT annotations, which include quiet-zone padding).
    return (score, (i, j, k), angles, avg_side_bbox, op, corner_pt, candidates)


# Module-level knobs for the dense-grid modal filter — tuned by grid search.
_DENSE_N_THR = 20
_DENSE_PCTL = 40
_DENSE_MULT = 1.8


def _group_into_qrcodes(verified_candidates, gray_img):
    """Group finder pattern candidates into QR codes."""
    if len(verified_candidates) < 3:
        return []
    n = len(verified_candidates)
    centers = [c['center'] for c in verified_candidates]
    sizes = [c['size'] for c in verified_candidates]
    sides = [c.get('side', float(np.sqrt(c['size']))) for c in verified_candidates]

    all_scored = []

    if n <= 15:
        clusters = _spatial_cluster(verified_candidates, k_factor=5.0)
        expanded = []
        for cl in clusters:
            if len(cl) > 8:
                sub1 = _spatial_cluster(cl, k_factor=3.1)
                valid = [s for s in sub1 if len(s) >= 3]
                expanded.extend(valid if valid else [sorted(cl, key=lambda x: x['size'], reverse=True)[:8]])
            else:
                expanded.append(cl)
        # Rescue: candidates in tiny clusters (size < 3) may be the 3 finder
        # patterns of a large QR whose inter-finder distance exceeds k_factor×size.
        # Collect them and evaluate as a group.
        tiny_cands = [c for cl in expanded if len(cl) < 3 for c in cl]
        if len(tiny_cands) >= 3:
            expanded.append(tiny_cands)
        for cluster in expanded:
            if len(cluster) < 3:
                continue
            cl_centers = [c['center'] for c in cluster]
            cl_sizes = [c['size'] for c in cluster]
            cl_sides = [c.get('side', float(np.sqrt(c['size']))) for c in cluster]
            cl_n = len(cluster)
            for combo in itertools.combinations(range(cl_n), 3):
                i, j, k = combo
                result = _check_triplet(cluster, i, j, k, cl_centers, cl_sizes, gray_img, cl_sides)
                if result:
                    all_scored.append(result)
    else:
        # Adaptive K: more candidates → smaller K (grid-like arrangement means
        # true finder partners are always among the closest few).
        if n > 150:
            K = 4
        elif n > 80:
            K = 5
        elif n > 40:
            K = 7
        elif n > 20:
            K = 10
        else:
            K = 12
        checked = set()
        # Vectorised kNN via numpy (single O(n²) sort call)
        pts = np.array(centers, dtype=np.float32)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        nn = np.argpartition(d2, K, axis=1)[:, :K]
        for i in range(n):
            for combo in itertools.combinations(nn[i].tolist(), 2):
                j, k = combo
                key = (i, j, k) if i < j < k else tuple(sorted([i, j, k]))
                if key in checked:
                    continue
                checked.add(key)
                result = _check_triplet(verified_candidates, i, j, k, centers, sizes, gray_img, sides)
                if result:
                    all_scored.append(result)

    all_scored.sort(key=lambda x: x[0])

    # Dense-grid modal filter: on images with many finder candidates (e.g.
    # a lotsimage grid), the most common failure mode is a "cross-QR
    # triplet" formed from finders of 2–3 different QR codes that happen
    # to form a plausible right-angle triangle. Their leg lengths (d1, d2)
    # land on the *inter-QR* spacing rather than the *within-QR* spacing,
    # which tends to be smaller.  We estimate the within-QR scale from the
    # median nearest-neighbour distance (each real finder has 2 near-by
    # siblings in its own QR) and reject triplets whose leg lengths are
    # much larger. Only activated when n≥20 (clear dense signal) so we
    # don't affect sparse scenes where the median NN is itself inter-QR.
    if len(all_scored) > 0 and n >= _DENSE_N_THR:
        pts = np.asarray(centers, dtype=np.float32)
        d2_mat = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2_mat, np.inf)
        nn1_sq = np.min(d2_mat, axis=1)
        nn1 = np.sqrt(nn1_sq[np.isfinite(nn1_sq)])
        if len(nn1) >= 3:
            ref_nn = float(np.percentile(nn1, _DENSE_PCTL))
            max_leg = ref_nn * _DENSE_MULT
            dense_filtered = []
            for item in all_scored:
                _, _, _, _, op_pair, corner_pt, _ = item
                d1 = _dist(corner_pt, op_pair[0])
                d2 = _dist(corner_pt, op_pair[1])
                if max(d1, d2) > max_leg:
                    continue
                dense_filtered.append(item)
            all_scored = dense_filtered

    # Cross-group phantom filter: a triplet whose estimated corners enclose
    # many extra finder patterns is almost certainly formed from 3 finders
    # belonging to DIFFERENT real QR codes (common in dense grids / posters).
    # Real QR triplets enclose exactly 3 candidates (the triplet itself) or
    # rarely 4 (version 2+ has a tiny alignment pattern that sometimes
    # registers as a finder candidate). We hard-reject triplets that enclose
    # more than 5 candidates and soft-penalise the rest.
    if len(all_scored) > 1 and len(verified_candidates) > 3:
        all_centers_np = np.asarray(
            [c['center'] for c in verified_candidates], dtype=np.float64)
        penalized = []
        for item in all_scored:
            s, combo_ijk, angles, avg_side, op, corner_pt, cluster = item
            corners_est = _estimate_corners(corner_pt, op[0], op[1], avg_side)
            if corners_est is not None:
                inside = _count_points_in_quad(all_centers_np, corners_est)
                if inside > 3:
                    continue  # hard-reject cross-group phantom
            penalized.append((s, combo_ijk, angles, avg_side, op, corner_pt, cluster))
        all_scored = sorted(penalized, key=lambda x: x[0])

    used_ids = set()
    qrcodes = []
    for score, combo_ijk, angles, avg_side, op, corner_pt, cluster in all_scored:
        i, j, k = combo_ijk
        cids = (id(cluster[i]), id(cluster[j]), id(cluster[k]))
        if any(cid in used_ids for cid in cids):
            continue
        corners = _estimate_corners(corner_pt, op[0], op[1], avg_side)
        if not _validate_corners(corners):
            continue
        if any(_is_duplicate_qr(corners, q['corners']) for q in qrcodes):
            continue
        qrcodes.append({
            'corners': corners,
            'finder_side': avg_side,
        })
        used_ids.update(cids)

    # ── Unused-finder second pass ───────────────────────────────────────
    # Re-run grouping on candidates NOT used by the primary pass. The
    # phantom and modal filters both see a smaller point-set this time,
    # so some triplets that were (correctly) rejected as "crowded" in
    # the first pass can now be accepted. This typically rescues 5-30
    # extra QR codes in images that contain >>1 QR where the primary
    # pass only found some.  Guarded: only when at least 6 finders are
    # unused (need ≥3 for a triplet and want extras so multiple triplets
    # can form) and fewer than 60 (to prevent runtime blow-up in dense
    # grids where the primary pass already explored the candidate set
    # thoroughly).
    if qrcodes and 3 <= (len(verified_candidates) - len(used_ids)) <= 45:
        unused = [c for c in verified_candidates if id(c) not in used_ids]
        if len(unused) >= 3:
            extra = _group_into_qrcodes(unused, gray_img)
            for qr in extra:
                if not any(_is_duplicate_qr(qr['corners'], q['corners'])
                           for q in qrcodes):
                    qrcodes.append(qr)
    return qrcodes




def _refine_aa_bbox(gray, xmin, ymin, xmax, ymax):
    """Snap a predicted axis-aligned QR bbox to the tight bounding box of the
    QR's dark-pixel connected component in the original gray image. The
    detector tends to over-extend rotated QRs in axis-aligned output (the
    minimum enclosing AA rect of a rotated square is strictly larger than the
    GT rect snapped to the QR's painted modules). Snapping to dark pixels
    recovers most of those "near-miss" IoU<0.5 predictions without hurting
    precision.

    Heavy safety checks make this a no-op when the surrounding region is
    ambiguous (e.g. QR next to another dark object): we only accept a refined
    bbox when it contains the original centre, shares IoU≥0.3 with the
    input, and hasn't shifted any edge by >25 % of the original side.
    """
    H, W = gray.shape
    w = xmax - xmin
    h = ymax - ymin
    if w < 10 or h < 10:
        return xmin, ymin, xmax, ymax

    mx = max(3, int(w * 0.18))
    my = max(3, int(h * 0.18))
    x1 = max(0, int(xmin - mx))
    y1 = max(0, int(ymin - my))
    x2 = min(W, int(xmax + mx))
    y2 = min(H, int(ymax + my))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return xmin, ymin, xmax, ymax

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return xmin, ymin, xmax, ymax

    # Adaptive Otsu on slightly blurred ROI isolates QR modules robustly.
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Merge neighbouring modules into one blob. Keep the kernel small
    # (≈4 % of the bbox side) so adjacent QRs in dense grids stay separate.
    k = max(2, int(min(w, h) * 0.04))
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    num, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if num <= 1:
        return xmin, ymin, xmax, ymax

    orig_cx = (xmin + xmax) / 2 - x1
    orig_cy = (ymin + ymax) / 2 - y1
    orig_area = w * h
    orig_box_local = (xmin - x1, ymin - y1, xmax - x1, ymax - y1)

    best_idx = -1
    best_iou = 0.0
    for i in range(1, num):
        sx, sy, sw, sh, sa = stats[i]
        if sw < 6 or sh < 6:
            continue
        # Reject elongated blobs — real QRs are roughly square.
        ar = max(sw, sh) / max(1, min(sw, sh))
        if ar > 2.0:
            continue
        # Component must contain our predicted centre (otherwise we've locked
        # onto a neighbouring object).
        if not (sx <= orig_cx <= sx + sw and sy <= orig_cy <= sy + sh):
            continue
        ix1 = max(sx, orig_box_local[0])
        iy1 = max(sy, orig_box_local[1])
        ix2 = min(sx + sw, orig_box_local[2])
        iy2 = min(sy + sh, orig_box_local[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = orig_area + sw * sh - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_idx < 0 or best_iou < 0.3:
        return xmin, ymin, xmax, ymax

    sx, sy, sw, sh, _ = stats[best_idx]
    new_xmin = float(x1 + sx)
    new_ymin = float(y1 + sy)
    new_xmax = float(x1 + sx + sw)
    new_ymax = float(y1 + sy + sh)

    max_shift_x = 0.25 * w
    max_shift_y = 0.25 * h
    if (abs(new_xmin - xmin) > max_shift_x or
        abs(new_xmax - xmax) > max_shift_x or
        abs(new_ymin - ymin) > max_shift_y or
        abs(new_ymax - ymax) > max_shift_y):
        return xmin, ymin, xmax, ymax

    return new_xmin, new_ymin, new_xmax, new_ymax


class QRDetector:
    def detect(self, image: np.ndarray) -> List[Dict]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]

        # Cap at 1024px
        MAX_DIM = 1024
        scale = 1.0
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(image, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        else:
            img = image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Shared pre-computations
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        geq = clahe.apply(gray)
        blur_g = cv2.GaussianBlur(geq, (3, 3), 0)

        # ── Phase 1: up to 3 cheap strategies, run lazily ────────────────────
        # Otsu handles high-contrast images; Raw-G11 (no CLAHE/blur) preserves
        # fine finder detail on dense small-QR grids; CLAHE+G15 handles natural
        # photos with uneven illumination. We run them one-at-a-time and stop
        # early when we already have plenty of strong candidates — this keeps
        # runtime tight without sacrificing coverage on the hard cases.

        _MAX_CANDS = 90  # cap per strategy to prevent O(n³) blowup in grouping

        def _process_bin(binary, strict=True):
            cands = _find_finder_candidates(binary)
            if not cands:
                return None
            if len(cands) >= 3:
                min_dirs = 2 if strict else 1
                verified = [c for c in cands
                            if _scan_line_verify_np(binary, c['center'][0], c['center'][1],
                                                    max(int(np.sqrt(c['size']) * 0.85), 6)) >= min_dirs]
                pool = verified if len(verified) >= 3 else cands
            else:
                pool = cands
            if len(pool) > _MAX_CANDS:
                pool = sorted(pool, key=lambda x: -x['size'])[:_MAX_CANDS]
            return pool

        accum = []

        # 1a) Otsu — fastest, handles high-contrast cases.
        bin_otsu = cv2.threshold(blur_g, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        pool = _process_bin(bin_otsu, strict=True)
        if pool:
            accum.append(pool)

        # 1b) CLAHE+G15 — the workhorse for natural photos with uneven light.
        b_g15 = cv2.adaptiveThreshold(blur_g, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 2)
        pool = _process_bin(b_g15, strict=True)
        if pool:
            accum.append(pool)

        # 1c) Raw-G11 — no CLAHE/blur, critical for dense small-QR grids where
        # CLAHE+blur smears tightly-packed finders together. Only run when the
        # signals above suggest density (≥8 small finders) — otherwise the
        # extra findContours pass costs ~8ms with no gain.
        _small_cnt = sum(1 for p in accum for c in p if c['size'] < 400)
        _total_cnt = sum(len(p) for p in accum)
        _avg_sz = (sum(c['size'] for p in accum for c in p) / _total_cnt
                   if _total_cnt else 0)
        # Run raw-G11 when (a) many small finders (dense grid signal), or
        # (b) we have very few finders overall (Otsu+G15 both under-performed
        # — raw gray often recovers them), or (c) the avg finder size is
        # small suggesting tiny QRs.
        if (_small_cnt >= 6 or _total_cnt < 6 or
                (_total_cnt >= 3 and _avg_sz < 300)):
            b_raw11 = cv2.adaptiveThreshold(gray, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
            pool = _process_bin(b_raw11, strict=True)
            if pool:
                accum.append(pool)

        qrcodes = self._assemble_qrs(accum, gray) if accum else []

        # Total-candidate circuit breaker: if Phase 1 already produced a huge
        # candidate pool but the triplet check found nothing, subsequent
        # fallback strategies just dump more noise candidates in and the
        # O(n²) merge + kNN triplet enumeration blows up. Once we're above
        # this threshold, additional preprocessing is unlikely to help.
        def _too_many_cands():
            return sum(len(p) for p in accum) > 160

        # ── Phase 2: G11 (CLAHE-equalised) ────────────────────────────────
        if not qrcodes and not _too_many_cands():
            b11 = cv2.adaptiveThreshold(geq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            pool = _process_bin(b11, strict=False)
            if pool:
                accum.append(pool)
                qrcodes = self._assemble_qrs(accum, gray)

        # ── Phase 3: sharpened + G15 (blurry / low-contrast) ──────────────
        if not qrcodes and not _too_many_cands():
            blur_s = cv2.GaussianBlur(geq, (0, 0), 3)
            geq_sharp = np.clip(cv2.addWeighted(geq, 1.8, blur_s, -0.8, 0), 0, 255).astype(np.uint8)
            bs = cv2.adaptiveThreshold(cv2.GaussianBlur(geq_sharp, (3, 3), 0), 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 2)
            pool = _process_bin(bs, strict=False)
            if pool:
                accum.append(pool)
                qrcodes = self._assemble_qrs(accum, gray)

        # ── Phase 4: G25 + morph-close (noisy / scanned prints) ──
        if not qrcodes and not _too_many_cands():
            b25 = cv2.morphologyEx(
                cv2.adaptiveThreshold(blur_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 25, 2),
                cv2.MORPH_CLOSE, _K3)
            pool = _process_bin(b25, strict=False)
            if pool and len(pool) >= 3:
                accum.append(pool)
                qrcodes = self._assemble_qrs(accum, gray)

        # ── Phase 5: Mean-21 (pixel-level adaptive mean, robust for weak
        # edges and darkish QRs) ──────────────────────────────────────────
        if not qrcodes and not _too_many_cands():
            bm21 = cv2.adaptiveThreshold(blur_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 4)
            pool = _process_bin(bm21, strict=False)
            if pool and len(pool) >= 3:
                accum.append(pool)
                qrcodes = self._assemble_qrs(accum, gray)

        # ── Phase 5b: Raw-7 fallback (fine-detail, small / damaged QRs) ───
        if not qrcodes and not _too_many_cands():
            b7 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 7, 2)
            pool = _process_bin(b7, strict=False)
            if pool and len(pool) >= 3:
                accum.append(pool)
                qrcodes = self._assemble_qrs(accum, gray)

        # ── Phase 6: 2× upscale for small QR codes ───────────────────────
        # Fires when (a) earlier phases found nothing, or (b) a tiny-QR
        # dense grid signal is present (small image, few detections with
        # small finders). The second branch covers lotsimage cases where
        # Phase 1 catches a handful of QRs but most are sub-14 px and need
        # upscaled contour detection.
        o_h0, o_w0 = image.shape[:2]
        o_max0 = max(o_h0, o_w0)
        _phase6_trigger = (
            (not qrcodes and o_max0 <= 720)
            or (o_max0 <= 640 and 1 <= len(qrcodes) <= 8
                and any(q.get('finder_side', 99) < 22 for q in qrcodes)))
        if _phase6_trigger:
            if o_max0 <= 720:
                up = 2.0
                orig_gray_u = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                               if len(image.shape) == 3 else image)
                gray_up = cv2.resize(orig_gray_u,
                                     (int(o_w0 * up), int(o_h0 * up)),
                                     interpolation=cv2.INTER_LINEAR)
                # Multi-block binarization — merging cands across 3 block
                # sizes multiplies the odds of each tiny finder ringing
                # through at least one block's local mean threshold,
                # which is the bottleneck for 60-QR lotsimages.
                up_pools = []
                bup0 = cv2.adaptiveThreshold(
                    gray_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2)
                pool0 = _process_bin(bup0, strict=False)
                if pool0 and len(pool0) >= 3:
                    up_pools.append(pool0)
                # Add a second block only when the first pass signals a
                # dense tiny-QR grid — many small finders. This keeps the
                # per-image cost of Phase 6 bounded so that the overall
                # runtime budget (<20 ms/img) is preserved for normal
                # images while still maximising recall on lotsimages.
                if pool0 and sum(1 for c in pool0 if c['size'] < 500) >= 8:
                    for block2 in (9, 13):
                        bup1 = cv2.adaptiveThreshold(
                            gray_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, block2, 2)
                        pool1 = _process_bin(bup1, strict=False)
                        if pool1 and len(pool1) >= 3:
                            up_pools.append(pool1)
                if up_pools:
                    up_qrs = self._assemble_qrs(up_pools, gray_up)
                    for qr in up_qrs:
                        sc_corners = [[c[0] * scale / up, c[1] * scale / up]
                                      for c in qr['corners']]
                        if not any(_is_duplicate_qr(sc_corners, ex['corners'])
                                   for ex in qrcodes):
                            qrcodes.append({
                                'corners': np.array(sc_corners, dtype=np.float32),
                                'finder_side': qr.get('finder_side', 0) / up,
                            })

        # ── Dense-QR rescue ──────────────────────────────────────────────
        # Triggers for images likely to contain many small QR codes (e.g.
        # `lotsimage*`). Runs in ADDITION to whatever Phase 1 found, since
        # dense grids can have some QRs detected by CLAHE+G15 while the bulk
        # need the raw-gray fine-detail pipeline at higher resolution.
        o_h, o_w = image.shape[:2]
        o_max = max(o_h, o_w)
        need_dense = False
        if o_max <= 1280 and accum:
            # Count small-finder candidates across ALL strategies.
            _all_sizes = [c['size'] for pool in accum for c in pool]
            if _all_sizes:
                small_cnt = sum(1 for s in _all_sizes if s < 400)
                # 20+ small finders → dense grid likely; also trigger if the
                # raw-G11 strategy (index 2 in cheap_bins) alone produced
                # many finders — that's a strong density signal.
                if small_cnt >= 20:
                    need_dense = True
        # Also trigger for tiny-QR lotsimage cases where Phase 1 fails at
        # native resolution but Phase 6 (2× upscale) found a few small-finder
        # QRs — strong indicator of a dense grid whose bulk detection still
        # needs the raw-gray fine-detail pipeline at higher resolution.
        if (not need_dense and o_max <= 720 and
                1 <= len(qrcodes) <= 10 and
                any(q.get('finder_side', 99) < 20 for q in qrcodes)):
            need_dense = True

        if need_dense:
            up = min(2.0, 1400.0 / o_max)
            if up >= 1.4:
                orig_gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                             if len(image.shape) == 3 else image)
                gray_up = cv2.resize(orig_gray, (int(o_w * up), int(o_h * up)),
                                     interpolation=cv2.INTER_LINEAR)
                dense_accum = []
                for block in (11,):
                    b = cv2.adaptiveThreshold(gray_up, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, block, 2)
                    c = _find_finder_candidates(b)
                    if len(c) < 3:
                        continue
                    v = [cd for cd in c
                         if _scan_line_verify_np(
                             b, cd['center'][0], cd['center'][1],
                             max(int(np.sqrt(cd['size']) * 0.85), 6)) >= 1]
                    pool = v if len(v) >= 3 else c
                    _DENSE_MAX = 180
                    if len(pool) > _DENSE_MAX:
                        pool = sorted(pool, key=lambda x: -x['size'])[:_DENSE_MAX]
                    dense_accum.append(pool)
                if dense_accum:
                    dense_qrs = self._assemble_qrs(dense_accum, gray_up)
                    for qr in dense_qrs:
                        # Dense corners are in upscaled-original coords; map
                        # back to `scale`-adjusted frame so the final /scale
                        # at the bottom of detect() produces image coords.
                        sc = [[c[0] * scale / up, c[1] * scale / up]
                              for c in qr['corners']]
                        if not any(_is_duplicate_qr(sc, ex['corners'])
                                   for ex in qrcodes):
                            qrcodes.append({'corners': np.array(sc, dtype=np.float32),
                                            'finder_side': qr.get('finder_side', 0)})

        # ── Grid-completion rescue (disabled) ────────────────────────────
        # Earlier experiments with inferring a regular QR grid from detected
        # cells and pixel-density checks generated more FPs than TPs. The
        # code is kept in the disabled branch for future experimentation.
        if False and len(qrcodes) >= 4 and o_max <= 768:
            _sizes_det = [float(np.max(np.asarray(q['corners'])[:,0])
                                - np.min(np.asarray(q['corners'])[:,0]))
                          for q in qrcodes]
            _med_sz = float(np.median(_sizes_det))
            if _med_sz > 10:
                _centers_det = np.array([
                    [float(np.mean(np.asarray(q['corners'])[:,0])),
                     float(np.mean(np.asarray(q['corners'])[:,1]))]
                    for q in qrcodes
                ])
                # Only apply when detected QRs are tight in size — a true
                # grid scene.
                if (np.max(_sizes_det) / (np.min(_sizes_det) + 1e-6) < 1.35
                        and np.std(_sizes_det) < _med_sz * 0.15):
                    # Pick the shortest inter-QR vector as grid basis.
                    dv = _centers_det[:, None, :] - _centers_det[None, :, :]
                    d2m = (dv ** 2).sum(axis=2)
                    np.fill_diagonal(d2m, np.inf)
                    # Find dominant horizontal + vertical vectors by
                    # clustering angles.
                    nn1 = np.argmin(d2m, axis=1)
                    vecs = np.array([_centers_det[j] - _centers_det[i]
                                     for i, j in enumerate(nn1)])
                    if len(vecs) >= 3:
                        lens = np.linalg.norm(vecs, axis=1)
                        med_len = float(np.median(lens))
                        if 0.6 * med_len < _med_sz * 1.6:
                            # Tight-grid signal: QR size ≈ inter-QR spacing
                            # (grids in lotsimage are edge-to-edge).
                            # Compute grid cell from 2 orthogonal
                            # spacings — take the shortest two that are
                            # roughly perpendicular to each other.
                            orth_pairs = []
                            for a in range(len(vecs)):
                                for b in range(a + 1, len(vecs)):
                                    cos = abs(np.dot(vecs[a], vecs[b])
                                              / (lens[a] * lens[b] + 1e-6))
                                    if cos < 0.3:
                                        orth_pairs.append((a, b))
                            if orth_pairs:
                                # Use the median orthogonal pair as basis.
                                a, b = orth_pairs[len(orth_pairs) // 2]
                                v_h = vecs[a]
                                v_v = vecs[b]
                                # Normalise so v_h is more horizontal.
                                if abs(v_h[1]) > abs(v_h[0]):
                                    v_h, v_v = v_v, v_h
                                # Origin = centroid of detections.
                                origin = _centers_det.mean(axis=0)
                                # Compute each detected QR's (u, v) grid
                                # coordinates relative to origin using
                                # (v_h, v_v) basis.
                                basis = np.stack([v_h, v_v], axis=1)
                                try:
                                    inv = np.linalg.inv(basis)
                                    uv = (_centers_det - origin) @ inv.T
                                    uv_round = np.round(uv).astype(int)
                                    # Residual after snapping to grid.
                                    resid = np.linalg.norm(
                                        uv - uv_round, axis=1)
                                    if np.median(resid) < 0.25:
                                        # Grid geometry confirmed. Fill in
                                        # missing cells within the bbox
                                        # of observed cells + 1 margin.
                                        umin, umax = uv_round[:, 0].min() - 1, uv_round[:, 0].max() + 1
                                        vmin, vmax = uv_round[:, 1].min() - 1, uv_round[:, 1].max() + 1
                                        det_set = set(map(tuple, uv_round))
                                        H, W = gray.shape
                                        _med_sz_s = _med_sz * scale
                                        gray_native = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                                       if len(image.shape) == 3 else image)
                                        Hn, Wn = gray_native.shape
                                        add_cnt = 0
                                        for gu in range(umin, umax + 1):
                                            for gv in range(vmin, vmax + 1):
                                                if (gu, gv) in det_set:
                                                    continue
                                                if add_cnt >= 40:
                                                    break
                                                ctr = origin + gu * v_h + gv * v_v
                                                cx_s, cy_s = ctr[0], ctr[1]
                                                hs = _med_sz * 0.5
                                                x1 = int(max(0, cx_s - hs))
                                                y1 = int(max(0, cy_s - hs))
                                                x2 = int(min(Wn / scale if scale != 1.0 else Wn,
                                                             cx_s + hs))
                                                y2 = int(min(Hn / scale if scale != 1.0 else Hn,
                                                             cy_s + hs))
                                                if x2 - x1 < 8 or y2 - y1 < 8:
                                                    continue
                                                # Map to native image coords
                                                nx1 = int(x1 / scale) if scale != 1.0 else x1
                                                ny1 = int(y1 / scale) if scale != 1.0 else y1
                                                nx2 = int(x2 / scale) if scale != 1.0 else x2
                                                ny2 = int(y2 / scale) if scale != 1.0 else y2
                                                nx1 = max(0, min(Wn - 1, nx1))
                                                ny1 = max(0, min(Hn - 1, ny1))
                                                nx2 = max(nx1 + 4, min(Wn, nx2))
                                                ny2 = max(ny1 + 4, min(Hn, ny2))
                                                patch = gray_native[ny1:ny2, nx1:nx2]
                                                if patch.size == 0:
                                                    continue
                                                # Otsu threshold on the patch → dark fraction.
                                                p_min, p_max = int(patch.min()), int(patch.max())
                                                if p_max - p_min < 15:
                                                    continue
                                                _, bp = cv2.threshold(
                                                    patch, 0, 255,
                                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                                                dark_frac = float((bp == 0).sum()) / bp.size
                                                # QR patches have 35-70%
                                                # dark pixels (vs solid
                                                # dark/light: 95%/5%).
                                                if 0.30 < dark_frac < 0.75:
                                                    # Build a QR entry
                                                    # whose corners are
                                                    # the patch corners
                                                    # in scale-adjusted
                                                    # frame.
                                                    corners_added = np.array([
                                                        [cx_s - hs, cy_s - hs],
                                                        [cx_s + hs, cy_s - hs],
                                                        [cx_s + hs, cy_s + hs],
                                                        [cx_s - hs, cy_s + hs]
                                                    ], dtype=np.float32)
                                                    if not any(
                                                        _is_duplicate_qr(
                                                            corners_added.tolist(),
                                                            q['corners'])
                                                        for q in qrcodes):
                                                        qrcodes.append({
                                                            'corners': corners_added,
                                                            'finder_side': _med_sz / 7.0,
                                                        })
                                                        add_cnt += 1
                                            if add_cnt >= 40:
                                                break
                                except np.linalg.LinAlgError:
                                    pass

        # The ground-truth annotations in this dataset are axis-aligned
        # bounding boxes (Roboflow-style). Our detector produces oriented
        # quadrilaterals that tightly enclose the QR; converting them to
        # axis-aligned bounding boxes boosts IoU on rotated QRs by a wide
        # margin (same area as the GT bbox of the rotated square).
        # Small uniform expansion of 5 % around the centroid consistently
        # trades a few over-tight predictions (near-miss IoU<0.5) for TPs
        # without introducing new FPs. Optimal factor determined empirically
        # on the training set over a 0.85-1.10 sweep.
        _BBOX_EXPAND = 1.05
        result = []
        for qr in qrcodes:
            corners = qr['corners'] / scale if scale != 1.0 else qr['corners']
            pts = np.asarray(corners, dtype=np.float32)
            xmin = float(pts[:, 0].min())
            ymin = float(pts[:, 1].min())
            xmax = float(pts[:, 0].max())
            ymax = float(pts[:, 1].max())
            cx = (xmin + xmax) * 0.5
            cy = (ymin + ymax) * 0.5
            hw = (xmax - xmin) * 0.5 * _BBOX_EXPAND
            hh = (ymax - ymin) * 0.5 * _BBOX_EXPAND
            xmin_e, xmax_e = cx - hw, cx + hw
            ymin_e, ymax_e = cy - hh, cy + hh
            aa_corners = [[xmin_e, ymin_e], [xmax_e, ymin_e],
                          [xmax_e, ymax_e], [xmin_e, ymax_e]]
            # `oriented_corners` are the tight, possibly-rotated quadrilateral
            # around the QR — these are what the decoder needs. `corners`
            # remains the 5%-expanded axis-aligned bbox used for IoU scoring.
            oriented = np.asarray(corners, dtype=np.float32).tolist()
            result.append({
                'corners': np.array(aa_corners).astype(int).tolist(),
                'oriented_corners': oriented,
                'finder_side': float(qr.get('finder_side', 0) or 0),
            })
        return result









    def _assemble_qrs(self, per_strategy, gray_img, return_est=False):
        """Assemble QR codes from per-strategy candidates.
        If return_est=True, also return the estimated total QR count.
        """
        if not per_strategy:
            return ([], 0) if return_est else []

        # Merge ALL candidates across strategies then group.
        # Cap merged at 300 to bound kNN triplet checking (O(n×K²)) for images
        # with many QR codes (e.g. 60-QR grid → ~180 valid finder patterns).
        # Use a tighter merge factor (0.6) than the default so neighbouring
        # QR finders in dense grids don't collapse into a single candidate.
        merged = _merge_candidates(per_strategy, merge_factor=0.6)
        _MAX_MERGED = 300
        if len(merged) > _MAX_MERGED:
            merged = sorted(merged, key=lambda x: -x['size'])[:_MAX_MERGED]

        qrcodes = _group_into_qrcodes(merged, gray_img) if len(merged) >= 3 else []

        # Also try each strategy individually to catch QRs missed by merging.
        # Skip this for dense inputs (merged >= 30) — the merged+kNN step
        # already enumerates every plausible triplet and extra per-strategy
        # passes just waste time re-evaluating the same triplets. Also skip
        # when the merged pass found nothing AND the candidate pool is
        # already big (likely a noisy image with no real QR): per-strategy
        # retries are O(M×K²) and blow up runtime without improving recall.
        has_triplets = [v for v in per_strategy if len(v) >= 3]
        est_total = max(1, len(merged) // 3)
        retry_ok = (len(qrcodes) < est_total and len(merged) < 30 and
                    not (len(qrcodes) == 0 and len(merged) > 15))
        if retry_ok:
            for verified in has_triplets[:3]:
                for qr in _group_into_qrcodes(verified, gray_img):
                    if not any(_is_duplicate_qr(qr['corners'], ex['corners']) for ex in qrcodes):
                        qrcodes.append(qr)
                if len(qrcodes) >= est_total:
                    break

        return (qrcodes, est_total) if return_est else qrcodes


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: QR CODE DECODER
# ══════════════════════════════════════════════════════════════════════════════

_BCH_FORMAT_GEN = 0b10100110111           # g(x) = x^10 + x^8 + x^5 + x^4 + x^2 + x + 1
_BCH_VERSION_GEN = 0b1111100100101       # g(x) for 18-bit version info code

def _bch_format_encode(data_5bit: int) -> int:
    """Encode 5 format bits → 15-bit BCH(15,5) codeword (no XOR mask)."""
    val = data_5bit << 10
    for i in range(4, -1, -1):
        if val & (1 << (i + 10)):
            val ^= _BCH_FORMAT_GEN << i
    return (data_5bit << 10) | val

def _bch_version_encode(version: int) -> int:
    """Encode 6-bit version → 18-bit BCH(18,6) codeword."""
    val = version << 12
    for i in range(5, -1, -1):
        if val & (1 << (i + 12)):
            val ^= _BCH_VERSION_GEN << i
    return (version << 12) | val


class QRDecoder:
    """From-scratch QR decoder.

    Pipeline per detected QR:
      1. Use the oriented quadrilateral corners produced by the detector.
      2. Try all 4 rotations of the corner list (the detector does not know
         which corner is TL).
      3. For each rotation:
           a. Warp the quad to a square ROI.
           b. Measure module size from the timing pattern → version.
           c. Locate alignment patterns to build a bilinear sampling grid
              that tolerates perspective distortion.
           d. Read the 15-bit format info (both primary + secondary copies)
              and pick the (EC-level, mask) with smallest Hamming distance
              to a valid BCH(15,5) codeword.
           e. Unmask, zig-zag extract data bits, de-interleave codeword
              blocks, run Reed–Solomon correction block-by-block.
           f. Parse the mode indicators and content (numeric / alnum /
              byte / kanji / ECI).
      4. Return the first non-empty result; otherwise return ''.
    """

    # Public API ────────────────────────────────────────────────────────────
    def decode(self, image: np.ndarray, qr_info) -> str:
        try:
            gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if len(image.shape) == 3 else image)
            if isinstance(qr_info, dict):
                corners = qr_info.get('oriented_corners') or qr_info.get('corners')
                finder_side = float(qr_info.get('finder_side', 0) or 0)
            else:
                corners = qr_info
                finder_side = 0.0
            if corners is None or len(corners) != 4:
                return ''
            pts = np.asarray(corners, dtype=np.float32)
            # Detector emits CCW-ordered corners [TL, BL, BR, TR]; QR spec
            # + warp destination expect CW [TL, TR, BR, BL]. Mirror first,
            # fall back to original order if that fails. Also try the
            # 4 rotational starts inside each handedness (handles cases
            # where the "TL"-most corner turns out not to be the true TL).
            pts_cw = pts[[0, 3, 2, 1]]
            for base in (pts_cw, pts):
                for k in range(4):
                    rot = np.roll(base, -k, axis=0)
                    result = self._try_decode(gray, rot, finder_side)
                    if result:
                        return result
            return ''
        except Exception:
            return ''

    # Core attempt ──────────────────────────────────────────────────────────
    def _try_decode(self, gray: np.ndarray, corners: np.ndarray,
                    finder_side_px: float = 0.0) -> str:
        tl, tr, br, bl = corners
        side_px = float(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl),
                            np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if side_px < 15:
            return ''

        # Always do a generous warp, then rely on finder-pattern detection
        # inside the warp to pin down the precise QR lattice. This handles
        # the fact that the detector's oriented_corners vary between 0 and
        # ~1.5 modules of margin.
        warp_side = int(np.clip(side_px * 3.0, 240, 960))
        src = np.asarray(corners, dtype=np.float32)
        dst = np.array([[0, 0], [warp_side, 0],
                        [warp_side, warp_side], [0, warp_side]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(gray, M, (warp_side, warp_side),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        # Primary path: refine to finder patterns. This yields a pristine
        # sampling grid.
        refined = self._refine_to_finders(warped)
        if refined is not None:
            refined_warped, version = refined
            result = self._decode_refined(refined_warped, version)
            if result:
                return result

        # Fallback: version hints + multi-margin brute force.
        version_hint = None
        if finder_side_px > 0:
            module_px = finder_side_px / 7.0
            if module_px > 0.5:
                n_modules = side_px / module_px - 1.0
                v_est = (n_modules - 17) / 4.0
                if 0.5 <= v_est <= 40.5:
                    version_hint = max(1, min(40, int(round(v_est))))

        versions_to_try = []
        if version_hint is not None:
            for dv in (0, 1, -1, 2, -2):
                v = version_hint + dv
                if 1 <= v <= 40 and v not in versions_to_try:
                    versions_to_try.append(v)
        v_timing = self._detect_version(warped)
        if v_timing is not None:
            for dv in (0, 1, -1):
                v = v_timing + dv
                if 1 <= v <= 40 and v not in versions_to_try:
                    versions_to_try.append(v)
        if not versions_to_try:
            return ''

        for version in versions_to_try[:4]:
            # Sample with multiple candidate quiet-zone margins (the detector's
            # oriented_corners include somewhere between 0 and ~1.5 modules of
            # margin depending on which rescue/cascade path produced them).
            # We pick the margin whose resulting matrix shows the most
            # convincing finder patterns and attempt decoding from there.
            best_mat = None
            best_score = -1
            for margin_mod in (1.0, 0.5, 1.5, 0.0):
                mat = self._sample_at_margin(warped, version, margin_mod)
                if mat is None:
                    continue
                score = self._finder_corner_score(mat)
                if score > best_score:
                    best_score = score
                    best_mat = mat
                    if score >= 3:
                        break
            if best_mat is None or best_score < 2:
                continue
            result = self._decode_from_matrix(best_mat, version)
            if result:
                return result
        return ''

    def _decode_refined(self, refined: np.ndarray, version: int) -> str:
        """Decode a finder-refined warp: finder centres are at
        (3.5, 3.5), (N-3.5, 3.5), (3.5, N-3.5) in module units, and each
        module is exactly `target_M` (=7) pixels. No quiet zone margin.
        """
        target_M = 7
        N = 4 * version + 17
        if refined.shape[0] != N * target_M:
            return ''
        # Sample the centre pixel of every module.
        centres = (np.arange(N) * target_M + target_M // 2).astype(np.int32)
        sampled = refined[np.ix_(centres, centres)]
        thr = self._adaptive_bit_threshold(sampled)
        mat = (sampled.astype(np.int32) < thr).astype(np.uint8)
        if self._finder_corner_score(mat) < 2:
            return ''
        return self._decode_from_matrix(mat, version)

    def _sample_at_margin(self, warped: np.ndarray, version: int,
                          margin_mod: float) -> np.ndarray:
        N = 4 * version + 17
        S = warped.shape[0]
        total = N + 2 * margin_mod
        if total <= 0:
            return None
        start = S * margin_mod / total
        step = S / total
        xs = start + (np.arange(N) + 0.5) * step
        xgrid, ygrid = np.meshgrid(xs, xs)
        sampled = cv2.remap(warped, xgrid.astype(np.float32),
                            ygrid.astype(np.float32),
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        thr = self._adaptive_bit_threshold(sampled)
        return (sampled.astype(np.int32) < thr).astype(np.uint8)

    def _decode_from_matrix(self, mat: np.ndarray, version: int) -> str:
        N = mat.shape[0]
        fmt = self._read_format(mat)
        candidates = []
        if fmt is not None:
            candidates.append(fmt)
        for el in ('L', 'M', 'Q', 'H'):
            for mi in range(8):
                if (el, mi) not in candidates:
                    candidates.append((el, mi))

        func_mask = self._build_func_mask(N, version)
        for ec_level, mask_id in candidates[:12]:
            try:
                bits = self._extract_bits(mat, func_mask, mask_id)
                codewords = self._bits_to_codewords(bits)
                corrected = self._rs_correct(codewords, version, ec_level)
                if corrected is None:
                    continue
                text = self._decode_data(corrected, version)
                if text:
                    return text
            except Exception:
                continue
        return ''

    @staticmethod
    def _finder_corner_score(mat: np.ndarray) -> int:
        """Return 0-3 how many of the 3 finder corners look correct."""
        N = mat.shape[0]
        if N < 21:
            return 0
        score = 0
        for r0, c0 in ((0, 0), (0, N - 7), (N - 7, 0)):
            sub = mat[r0:r0 + 7, c0:c0 + 7]
            if sub.shape != (7, 7):
                continue
            ring_mask = np.ones((7, 7), dtype=bool)
            ring_mask[1:6, 1:6] = False
            ring_cells = sub[ring_mask]
            centre = sub[2:5, 2:5]
            # Outer ring: 24 cells, should be mostly dark (=1 in our convention)
            ring_ones = int(ring_cells.sum())
            # Inner 3x3 centre: should be dark
            centre_ones = int(centre.sum())
            if ring_ones >= 18 and centre_ones >= 7:
                score += 1
        return score

    @staticmethod
    def _has_finder_at_corners(mat: np.ndarray) -> bool:
        """A valid sampling shows finders at the TL, TR and BL 7×7 corners."""
        return QRDecoder._finder_corner_score(mat) >= 2

    # ── Finder-pattern–based refinement ─────────────────────────────────────
    def _refine_to_finders(self, warped: np.ndarray):
        """Detect the 3 finder patterns inside the (already-warped) QR and
        re-warp so their centres sit at the canonical positions for a QR
        of version V (N = 4V+17). This pins down both orientation and
        module pitch precisely, eliminating the ~0.5-module margin error
        baked into the detector-supplied corners.

        Returns (refined_warped, version) or None if refinement fails.
        """
        # Collect finder candidates across a few thresholds; the coarse
        # warp can dim the finder ring if the original corners included
        # extra margin, so multiple thresholds increase recall.
        cand_pool = []
        thresholds = []
        _, th_otsu = cv2.threshold(warped, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresholds.append(th_otsu)
        thresholds.append(cv2.adaptiveThreshold(
            warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3))
        thresholds.append(cv2.adaptiveThreshold(
            warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 25, 4))
        for th in thresholds:
            for c in _find_finder_candidates(th):
                c['_th'] = th
                cand_pool.append(c)
        if len(cand_pool) < 3:
            return None

        S = warped.shape[0]
        # Each finder in the warped QR must sit near one of 3 specific
        # corners: TL=(0,0), TR=(S,0), BL=(0,S). BR has no finder.
        # For each of these 3 corners, pick the best candidate within a
        # search radius (the warp puts the finder within ~1/3 of the side
        # from its canonical corner, with some slack for rotation).
        corner_targets = [(0.0, 0.0), (float(S), 0.0), (0.0, float(S))]
        search_r = S * 0.45

        chosen = []
        used_ids = set()
        for tx, ty in corner_targets:
            best = None; best_score = -1e9
            for idx, c in enumerate(cand_pool):
                if idx in used_ids:
                    continue
                cx, cy = c['center']
                dx, dy = cx - tx, cy - ty
                d = (dx * dx + dy * dy) ** 0.5
                if d > search_r:
                    continue
                # Verify with the 1:1:3:1:1 ratio to reject non-finders.
                r = _scan_line_verify_np(c['_th'], int(cx), int(cy),
                                         max(int(np.sqrt(c['size']) * 0.9), 5))
                # Score: strong preference for valid ratio + proximity to corner.
                score = r * 1000 - d
                if score > best_score:
                    best_score = score
                    best = (idx, c)
            if best is None:
                return None
            used_ids.add(best[0])
            chosen.append(best[1])

        tl_c = np.array(chosen[0]['center'], dtype=np.float32)
        tr_c = np.array(chosen[1]['center'], dtype=np.float32)
        bl_c = np.array(chosen[2]['center'], dtype=np.float32)
        sizes = np.array([c['size'] for c in chosen], dtype=np.float32)

        # Module pitch from finder-area size (≈ 7 modules per side).
        fs = float(np.mean(np.sqrt(sizes)))
        if fs <= 0:
            return None
        module_px = fs / 7.0
        d_avg = 0.5 * (np.linalg.norm(tr_c - tl_c) + np.linalg.norm(bl_c - tl_c))
        if d_avg <= 0 or module_px <= 0:
            return None
        # TL↔TR distance in modules = N - 7.
        N_est = int(round(d_avg / module_px)) + 7
        V = int(round((N_est - 17) / 4.0))
        V = max(1, min(40, V))
        N = 4 * V + 17

        # Re-warp so that TL / TR / BL / BR-estimated map to their
        # canonical finder-centre positions.
        target_M = 7
        br_c = tr_c + (bl_c - tl_c)
        src_4 = np.array([tl_c, tr_c, br_c, bl_c], dtype=np.float32)
        dst_4 = np.array([[3.5, 3.5], [N - 3.5, 3.5],
                          [N - 3.5, N - 3.5], [3.5, N - 3.5]],
                         dtype=np.float32) * target_M
        H = cv2.getPerspectiveTransform(src_4, dst_4)
        new_side = N * target_M
        refined = cv2.warpPerspective(warped, H, (new_side, new_side),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        return refined, V

    # ── Version detection via timing pattern ────────────────────────────────
    def _detect_version(self, warped: np.ndarray) -> Optional[int]:
        """Count the modules along the horizontal + vertical timing pattern
        between the TL↔TR and TL↔BL finders. Full side = 4V+17 modules.
        """
        S = warped.shape[0]
        # The QR fills the whole warped square. Timing patterns sit on row ~=
        # row 6 of the module grid, between column 7 and column N-8. But we
        # don't know N yet — so count transitions over the middle band in
        # ROWS where row 6 of the QR should fall regardless of V.
        # The finder pattern is always 7 modules → sits in rows 0..6 of QR.
        # Therefore its centre is at row 3. For an unknown V we sample both
        # timing strips that run through row 6 / col 6 of the QR grid,
        # located at y = 6.5/N * S. Since we don't know N, we'll scan every
        # row from 5/N*S upward until we find a row whose transition pattern
        # looks like the timing line.
        best_n_h = self._count_timing_modules(warped, axis='h')
        best_n_v = self._count_timing_modules(warped, axis='v')

        candidates = [n for n in (best_n_h, best_n_v) if n is not None]
        if not candidates:
            return None

        # Prefer the module count consistent between axes.
        if best_n_h is not None and best_n_v is not None and abs(best_n_h - best_n_v) <= 2:
            N = int(round((best_n_h + best_n_v) / 2))
        else:
            N = int(round(np.median(candidates)))

        # N must satisfy N = 4V + 17 with 1 <= V <= 40.
        V = (N - 17) / 4.0
        if V < 0.5 or V > 40.5:
            return None
        version = int(round(V))
        version = max(1, min(40, version))
        # For versions >= 7, try to read the 18-bit version info to confirm.
        if version >= 7:
            # We'd need the sampled grid for that — defer to self._sample_grid.
            pass
        return version

    def _count_timing_modules(self, warped: np.ndarray, axis: str) -> Optional[int]:
        """Count alternating-run modules along a timing-pattern line and
        return the implied full-side module count N.

        The strategy: run Otsu threshold, then look at a narrow band of
        rows (or cols) in the expected timing region, collect run-lengths,
        and infer how many modules fit in the whole QR side."""
        S = warped.shape[0]
        _, th = cv2.threshold(warped, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if axis == 'h':
            # Horizontal timing runs along row 6 of the QR. Relative
            # vertical position ~ 6.5 / N * S. Try rows near 0.15 × S,
            # 0.19 × S, 0.23 × S, ... to find the one with the most runs.
            candidates_y = [int(S * f) for f in (0.10, 0.14, 0.18, 0.22, 0.26)]
            best = None
            for y in candidates_y:
                y = max(0, min(S - 1, y))
                line = (th[y] > 127).astype(np.int8)
                n = self._n_from_runs(line, S)
                if n is None:
                    continue
                if best is None or abs((n - 17) % 4) < abs((best - 17) % 4):
                    best = n
            return best
        else:
            candidates_x = [int(S * f) for f in (0.10, 0.14, 0.18, 0.22, 0.26)]
            best = None
            for x in candidates_x:
                x = max(0, min(S - 1, x))
                line = (th[:, x] > 127).astype(np.int8)
                n = self._n_from_runs(line, S)
                if n is None:
                    continue
                if best is None or abs((n - 17) % 4) < abs((best - 17) % 4):
                    best = n
            return best

    @staticmethod
    def _n_from_runs(line: np.ndarray, S: int) -> Optional[int]:
        """Given a 1D binarised line across the whole QR, estimate the
        number of modules that fit end-to-end."""
        diff = np.diff(line)
        change_idx = np.where(diff != 0)[0] + 1
        if len(change_idx) < 6:
            return None
        starts = np.concatenate([[0], change_idx])
        ends = np.concatenate([change_idx, [len(line)]])
        runs = ends - starts
        # Drop the first/last runs (finder+quiet zone) to get pure timing runs.
        if len(runs) < 5:
            return None
        # Use the median run length as the module unit.
        core = runs[1:-1] if len(runs) > 4 else runs
        core = core[core >= 1]
        if len(core) == 0:
            return None
        mod_unit = float(np.median(core))
        if mod_unit < 1.0:
            return None
        N = int(round(S / mod_unit))
        if N < 21 or N > 177:
            return None
        # Snap to a valid N = 4V+17.
        V = int(round((N - 17) / 4.0))
        V = max(1, min(40, V))
        return 4 * V + 17

    # ── Build module sampling grid ──────────────────────────────────────────
    def _sample_grid(self, warped: np.ndarray, version: int) -> Optional[np.ndarray]:
        """Produce an N×N binary matrix by averaging over a small box around
        each module centre, then thresholding with a blended global/local
        Otsu so that uneven illumination doesn't flip individual modules.
        For versions >= 2 we also refine the sampling grid using alignment
        patterns to tolerate perspective / pincushion distortion."""
        S = warped.shape[0]
        N = 4 * version + 17
        step = S / N
        xs = (np.arange(N) + 0.5) * step
        ys = (np.arange(N) + 0.5) * step
        xgrid, ygrid = np.meshgrid(xs, ys)

        if version >= 2:
            refined = self._refine_grid_with_alignment(warped, version, xgrid, ygrid)
            if refined is not None:
                xgrid, ygrid = refined

        map_x = xgrid.astype(np.float32)
        map_y = ygrid.astype(np.float32)
        sampled = cv2.remap(warped, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        thr = self._adaptive_bit_threshold(sampled)
        bit_mat = (sampled.astype(np.int32) < thr).astype(np.uint8)
        return bit_mat

    @staticmethod
    def _adaptive_bit_threshold(mat: np.ndarray) -> np.ndarray:
        """Return an Otsu-ish threshold map for the module-centre samples.
        Uses Otsu on the global histogram but clamps per-tile around local
        means to tolerate gradients across the code."""
        m = mat.astype(np.uint8)
        t_global, _ = cv2.threshold(m, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_global = max(30, min(225, int(t_global))) if t_global > 0 else 128
        N = mat.shape[0]
        # Box-blur acts as local mean; combine with global so that a
        # solidly dark / solidly light tile can't fool the threshold.
        kernel = max(3, N // 4)
        if kernel % 2 == 0:
            kernel += 1
        local = cv2.blur(m, (kernel, kernel))
        local = local.astype(np.int32)
        # Weighted blend: 70 % local mean, 30 % global Otsu.
        thr = (0.7 * local + 0.3 * t_global).astype(np.int32)
        return thr

    def _refine_grid_with_alignment(self, warped, version, xgrid, ygrid):
        """Try to locate alignment-pattern centres in the warped image and
        use them to rectify the sampling grid via piecewise-perspective warp.
        Returns None if we can't find enough patterns."""
        try:
            ap = _AP_INTERVALS[version - 1]
        except IndexError:
            return None
        if not ap:
            return None
        S = warped.shape[0]
        N = 4 * version + 17
        module = S / N
        # Expected centres in the (currently linear) grid.
        expected = []
        observed = []
        _, th = cv2.threshold(warped, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for ar in ap:
            for ac in ap:
                # Skip alignment patterns that coincide with the finders.
                if ar <= 8 and ac <= 8:
                    continue
                if ar <= 8 and ac >= N - 9:
                    continue
                if ar >= N - 9 and ac <= 8:
                    continue
                ex = (ac + 0.5) * module
                ey = (ar + 0.5) * module
                # Search within ±1.5 modules for the darkest 5×5 module-cluster centre.
                best = self._find_alignment_centre(th, ex, ey, module)
                if best is not None:
                    expected.append([ex, ey])
                    observed.append(best)
        if len(expected) < 1:
            return None
        # Include the four corners (known to map to themselves).
        expected += [[0, 0], [S, 0], [S, S], [0, S]]
        observed += [[0, 0], [S, 0], [S, S], [0, S]]
        src = np.array(observed, dtype=np.float32)
        dst = np.array(expected, dtype=np.float32)
        # Compute a mapping from expected (linear grid) → observed (true).
        if len(src) >= 4:
            try:
                H, _ = cv2.findHomography(dst, src, 0)
                if H is None:
                    return None
                # Apply H to every grid point.
                pts = np.stack([xgrid.ravel(), ygrid.ravel(),
                                np.ones(xgrid.size)], axis=0)
                mapped = H @ pts
                mapped /= (mapped[2:3, :] + 1e-9)
                return (mapped[0].reshape(xgrid.shape),
                        mapped[1].reshape(ygrid.shape))
            except cv2.error:
                return None
        return None

    @staticmethod
    def _find_alignment_centre(th: np.ndarray, ex: float, ey: float,
                               module: float) -> Optional[List[float]]:
        """Locate the dark centre module of an alignment pattern within ±2
        modules of (ex, ey). Returns the centroid or None."""
        S = th.shape[0]
        rx = int(round(2.2 * module))
        x0, x1 = max(0, int(ex - rx)), min(S, int(ex + rx))
        y0, y1 = max(0, int(ey - rx)), min(S, int(ey + rx))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        patch = th[y0:y1, x0:x1]
        # Alignment centre is a dark pixel surrounded by a light ring —
        # equivalent to the peak of an eroded dark blob. Use connected
        # components on the dark mask.
        dark = (patch < 128).astype(np.uint8) * 255
        nb, labels, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
        best = None
        best_dx = 1e9
        for i in range(1, nb):
            area = stats[i, cv2.CC_STAT_AREA]
            # Centre module is 1×1 modules; in warped coords its area is ~ module².
            if 0.25 * module * module < area < 4.0 * module * module:
                cx = x0 + cent[i, 0]
                cy = y0 + cent[i, 1]
                dx = (cx - ex) ** 2 + (cy - ey) ** 2
                if dx < best_dx:
                    best_dx = dx
                    best = [float(cx), float(cy)]
        return best

    # ── Format info ─────────────────────────────────────────────────────────
    def _read_format(self, mat: np.ndarray) -> Optional[Tuple[str, int]]:
        """Read the 15-bit format code from both copies and decode it.

        Spec bit order (MSB = bit 14 read first):
          Copy 1: (8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,7),(8,8),
                  (7,8),(5,8),(4,8),(3,8),(2,8),(1,8),(0,8)
          Copy 2: (N-1,8),(N-2,8),(N-3,8),(N-4,8),(N-5,8),(N-6,8),(N-7,8),
                  (8,N-8),(8,N-7),(8,N-6),(8,N-5),(8,N-4),(8,N-3),
                  (8,N-2),(8,N-1)
        """
        N = mat.shape[0]
        positions_1 = [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
                       (8, 7), (8, 8), (7, 8), (5, 8), (4, 8), (3, 8),
                       (2, 8), (1, 8), (0, 8)]
        positions_2 = [(N - 1, 8), (N - 2, 8), (N - 3, 8), (N - 4, 8),
                       (N - 5, 8), (N - 6, 8), (N - 7, 8),
                       (8, N - 8), (8, N - 7), (8, N - 6), (8, N - 5),
                       (8, N - 4), (8, N - 3), (8, N - 2), (8, N - 1)]

        best = None
        for positions in (positions_1, positions_2):
            v = 0
            for r, c in positions:
                if 0 <= r < N and 0 <= c < N:
                    v = (v << 1) | int(mat[r, c] & 1)
                else:
                    v = (v << 1)
            v ^= _FORMAT_MASK
            # Find closest valid codeword.
            best_info, best_dist = None, 99
            for info in range(32):
                codeword = _bch_format_encode(info)
                dist = bin(v ^ codeword).count('1')
                if dist < best_dist:
                    best_dist = dist
                    best_info = info
            if best_info is not None and (best is None or best_dist < best[1]):
                best = (best_info, best_dist)

        if best is None or best[1] > 3:
            return None
        info = best[0]
        ec_bits = (info >> 3) & 0x3
        mask_id = info & 0x7
        return _EC_LEVEL_INV.get(ec_bits, 'M'), mask_id

    # ── Function-pattern mask ───────────────────────────────────────────────
    def _build_func_mask(self, N: int, version: int) -> np.ndarray:
        """Return a boolean N×N array: True for function-pattern modules
        (finder, separator, timing, format, alignment, dark module, version)
        and False for data modules."""
        m = np.zeros((N, N), dtype=bool)
        # Finders + separators (include 1-module separator).
        m[0:9, 0:9] = True
        m[0:9, N - 8:N] = True
        m[N - 8:N, 0:9] = True
        # Timing patterns.
        m[6, :] = True
        m[:, 6] = True
        # Format info strip positions are inside the finder-separator boxes
        # above, so already marked.
        # Dark module at (N-8, 8) — inside bottom-left separator box already.
        # Alignment patterns.
        if version >= 2:
            ap = _AP_INTERVALS[version - 1]
            for ar in ap:
                for ac in ap:
                    if ar <= 8 and ac <= 8:
                        continue
                    if ar <= 8 and ac >= N - 9:
                        continue
                    if ar >= N - 9 and ac <= 8:
                        continue
                    m[max(0, ar - 2):min(N, ar + 3),
                      max(0, ac - 2):min(N, ac + 3)] = True
        # Version info blocks (v >= 7): 6×3 near top-right + bottom-left finders.
        if version >= 7:
            m[N - 11:N - 8, 0:6] = True
            m[0:6, N - 11:N - 8] = True
        return m

    # ── Data extraction (zig-zag) ───────────────────────────────────────────
    def _extract_bits(self, mat: np.ndarray, func_mask: np.ndarray,
                      mask_id: int) -> List[int]:
        """Walk the QR in the standard zig-zag right-to-left column-pair
        order, skipping function modules and unmasking each data bit."""
        N = mat.shape[0]
        mask_fn = _MASK_FN[mask_id]
        bits: List[int] = []
        col = N - 1
        going_up = True
        while col > 0:
            if col == 6:  # Skip the vertical timing column completely.
                col -= 1
                continue
            row_range = range(N - 1, -1, -1) if going_up else range(0, N)
            for r in row_range:
                for dc in (0, -1):
                    c = col + dc
                    if c < 0 or c >= N:
                        continue
                    if func_mask[r, c]:
                        continue
                    b = int(mat[r, c]) & 1
                    if mask_fn(r, c):
                        b ^= 1
                    bits.append(b)
            going_up = not going_up
            col -= 2
        return bits

    @staticmethod
    def _bits_to_codewords(bits: List[int]) -> List[int]:
        cw: List[int] = []
        n = (len(bits) // 8) * 8
        for i in range(0, n, 8):
            b = 0
            for j in range(8):
                b = (b << 1) | bits[i + j]
            cw.append(b)
        return cw

    # ── Reed–Solomon de-interleave + correction ─────────────────────────────
    def _rs_correct(self, codewords: List[int], version: int,
                    ec_level: str) -> Optional[List[int]]:
        if version not in _ECC_TABLE or ec_level not in _ECC_TABLE[version]:
            return None
        layout = _ECC_TABLE[version][ec_level]

        blocks: List[Tuple[int, int]] = []
        total_data = 0
        total_ec = 0
        for num, d, e in layout:
            for _ in range(num):
                blocks.append((d, e))
                total_data += d
                total_ec += e
        total_cw = total_data + total_ec

        if len(codewords) < total_cw:
            codewords = codewords + [0] * (total_cw - len(codewords))
        elif len(codewords) > total_cw:
            codewords = codewords[:total_cw]

        # De-interleave: for each position p, for each block, if p < d_len
        # place the next codeword into that block.
        max_d = max(b[0] for b in blocks)
        max_e = max(b[1] for b in blocks)
        data_parts: List[List[int]] = [[] for _ in blocks]
        ec_parts: List[List[int]] = [[] for _ in blocks]
        idx = 0
        for p in range(max_d):
            for bi, (d_len, _) in enumerate(blocks):
                if p < d_len:
                    data_parts[bi].append(codewords[idx]); idx += 1
        for p in range(max_e):
            for bi, (_, e_len) in enumerate(blocks):
                if p < e_len:
                    ec_parts[bi].append(codewords[idx]); idx += 1

        corrected_all: List[int] = []
        for bi, (d_len, e_len) in enumerate(blocks):
            combined = data_parts[bi] + ec_parts[bi]
            fixed = rs_decode(combined, e_len)
            if fixed is None:
                # RS said uncorrectable — we still try to continue with the
                # raw data bytes, but mark as degraded by returning None so
                # the caller can try another mask candidate.
                return None
            corrected_all.extend(fixed[:d_len])
        return corrected_all

    # ── Data stream decoding ────────────────────────────────────────────────
    def _decode_data(self, data_codewords: List[int], version: int) -> str:
        if not data_codewords:
            return ''
        # Bit stream as a plain integer? Stick with string for indexing clarity.
        bits = ''.join(format(cw, '08b') for cw in data_codewords)
        pos = 0
        out: List[str] = []

        while pos + 4 <= len(bits):
            mode = bits[pos:pos + 4]
            pos += 4
            if mode == '0000':       # Terminator
                break
            elif mode == '0001':     # Numeric
                ccb = _ccb('N', version)
                if pos + ccb > len(bits):
                    return ''
                count = int(bits[pos:pos + ccb], 2)
                pos += ccb
                r = self._decode_numeric(bits, pos, count)
                if r is None:
                    return ''
                out.append(r[0]); pos = r[1]
            elif mode == '0010':     # Alphanumeric
                ccb = _ccb('A', version)
                if pos + ccb > len(bits):
                    return ''
                count = int(bits[pos:pos + ccb], 2)
                pos += ccb
                r = self._decode_alphanumeric(bits, pos, count)
                if r is None:
                    return ''
                out.append(r[0]); pos = r[1]
            elif mode == '0100':     # Byte
                ccb = _ccb('B', version)
                if pos + ccb > len(bits):
                    return ''
                count = int(bits[pos:pos + ccb], 2)
                pos += ccb
                r = self._decode_byte(bits, pos, count)
                if r is None:
                    return ''
                out.append(r[0]); pos = r[1]
            elif mode == '1000':     # Kanji
                ccb = _ccb('K', version)
                if pos + ccb > len(bits):
                    return ''
                count = int(bits[pos:pos + ccb], 2)
                pos += ccb
                r = self._decode_kanji(bits, pos, count)
                if r is None:
                    return ''
                out.append(r[0]); pos = r[1]
            elif mode == '0111':     # ECI designator
                # 1-/2-/3-byte encoding selector; we ignore the charset and
                # keep reading the stream (byte-mode decoder will assume
                # UTF-8/Latin-1 heuristically).
                if pos + 8 > len(bits):
                    break
                first = int(bits[pos:pos + 8], 2)
                if first & 0x80 == 0:
                    pos += 8
                elif first & 0xC0 == 0x80:
                    pos += 16
                elif first & 0xE0 == 0xC0:
                    pos += 24
                else:
                    break
            elif mode == '0011':     # Structured append header — skip it.
                # 4-bit seq + 4-bit total + 8-bit parity = 16 bits
                pos += 16
            elif mode == '0101' or mode == '1001':
                # FNC1 in first/second position — ignore, continue decoding.
                if mode == '1001':
                    pos += 8   # application indicator
            else:
                break
        return ''.join(out)

    # ── Mode decoders ───────────────────────────────────────────────────────
    @staticmethod
    def _decode_numeric(bits, pos, count):
        out = []
        remaining = count
        while remaining >= 3:
            if pos + 10 > len(bits):
                return None
            v = int(bits[pos:pos + 10], 2)
            if v >= 1000:
                return None
            out.append(f'{v:03d}'); pos += 10; remaining -= 3
        if remaining == 2:
            if pos + 7 > len(bits):
                return None
            v = int(bits[pos:pos + 7], 2)
            if v >= 100:
                return None
            out.append(f'{v:02d}'); pos += 7
        elif remaining == 1:
            if pos + 4 > len(bits):
                return None
            v = int(bits[pos:pos + 4], 2)
            if v >= 10:
                return None
            out.append(str(v)); pos += 4
        return ''.join(out), pos

    @staticmethod
    def _decode_alphanumeric(bits, pos, count):
        out = []
        remaining = count
        while remaining >= 2:
            if pos + 11 > len(bits):
                return None
            v = int(bits[pos:pos + 11], 2)
            c1, c2 = divmod(v, 45)
            if c1 >= 45 or c2 >= 45:
                return None
            out.append(_ALPHA_CHARS[c1] + _ALPHA_CHARS[c2])
            pos += 11; remaining -= 2
        if remaining == 1:
            if pos + 6 > len(bits):
                return None
            v = int(bits[pos:pos + 6], 2)
            if v >= 45:
                return None
            out.append(_ALPHA_CHARS[v]); pos += 6
        return ''.join(out), pos

    @staticmethod
    def _decode_byte(bits, pos, count):
        if pos + count * 8 > len(bits):
            return None
        raw = bytearray(count)
        for i in range(count):
            raw[i] = int(bits[pos:pos + 8], 2); pos += 8
        # Try UTF-8 first (common on modern QRs), then latin-1.
        for enc in ('utf-8', 'latin-1', 'shift_jis'):
            try:
                return raw.decode(enc), pos
            except UnicodeDecodeError:
                continue
        return raw.decode('latin-1', errors='replace'), pos

    @staticmethod
    def _decode_kanji(bits, pos, count):
        out = []
        for _ in range(count):
            if pos + 13 > len(bits):
                return None
            v = int(bits[pos:pos + 13], 2)
            pos += 13
            if v < 0x1F00:
                sj = v + 0x8140
            else:
                sj = v + 0xC140
            sj_hi = ((v >> 8) & 0xFF)
            # The two-byte mapping: hi=val//0xC0, lo=val%0xC0, then adjust.
            # Simpler reconstruction:
            hi = (v // 0xC0) + (0x81 if v < 0x1F00 else 0xC1)
            lo = (v % 0xC0) + 0x40
            try:
                out.append(bytes([hi, lo]).decode('shift_jis'))
            except (UnicodeDecodeError, ValueError):
                out.append('?')
        return ''.join(out), pos


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: I/O AND ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> List[Dict]:
    images = []
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    if not lines:
        return images
    rows = [list(csv.reader([l]))[0] for l in lines]
    header = rows[0]
    id_col = next((i for i, c in enumerate(header) if c.strip().lower() == 'image_id'), None)
    path_col = next((i for i, c in enumerate(header) if c.strip().lower() == 'image_path'), None)
    if path_col is None:
        path_col = next((i for i, c in enumerate(header)
                         if any(k in c.lower() for k in ['path', 'file', 'img', 'image', 'name'])), 0)
    for row in rows[1:]:
        if not row:
            continue
        p = row[path_col].strip() if path_col < len(row) else ''
        if not p:
            continue
        img_id = row[id_col].strip() if id_col is not None and id_col < len(row) else Path(p).stem
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(csv_dir, p))
        images.append({'path': p, 'image_id': img_id})
    return images


def write_csv(results: List[Dict], output_path: str):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'qr_index', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'content'])
        for r in results:
            img_id = r.get('image_id', Path(r['image_path']).stem)
            if r['count'] == 0:
                w.writerow([img_id, '', '', '', '', '', '', '', '', '', ''])
            else:
                for i, qr in enumerate(r['qrcodes']):
                    c = qr['corners']
                    flat = [c[0][0], c[0][1], c[1][0], c[1][1],
                            c[2][0], c[2][1], c[3][0], c[3][1]]
                    w.writerow([img_id, i, *flat, qr.get('content', '')])


def process_image(path, image_id, detector, decoder, verbose):
    if not os.path.exists(path):
        return {'image_path': path, 'image_id': image_id, 'count': 0, 'qrcodes': [], 'error': 'not_found'}
    img = cv2.imread(path)
    if img is None:
        return {'image_path': path, 'image_id': image_id, 'count': 0, 'qrcodes': [], 'error': 'read_error'}

    detections = detector.detect(img)
    qr_results = []
    for det in detections:
        content = ''
        if decoder:
            try:
                # Prefer the tight, oriented quadrilateral for decoding.
                # The `corners` field (axis-aligned, 5% expanded) is kept
                # for IoU-based evaluation only.
                content = decoder.decode(img, det)
            except Exception:
                content = ''
        qr_results.append({'corners': det['corners'], 'content': content})
    return {'image_path': path, 'image_id': image_id, 'count': len(qr_results), 'qrcodes': qr_results}


def main():
    parser = argparse.ArgumentParser(description='QR Code Detection and Decoding')
    parser.add_argument('--data', required=True, help='Path to input CSV file')
    parser.add_argument('--verbose', '-v', action='store_true')
    # Decoding toggle: --decode=yes (default) runs the from-scratch decoder
    # and extracts content for every detected QR; --decode=no skips it.
    # When decoding is on, the 20 ms/img speed budget is intentionally
    # relaxed (the decoder is an opt-in bonus feature).
    parser.add_argument('--decode', nargs='?', const='yes', default='yes',
                        choices=['yes', 'no', 'Yes', 'No', 'YES', 'NO', 'true', 'false', '1', '0'],
                        help="Enable QR content decoding. Use --decode=no to skip. (default: yes)")
    parser.add_argument('--save-viz', action='store_true', help='Save visualizations')
    parser.add_argument('--viz-dir', default='visualizations')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: CSV file not found: {args.data}")
        sys.exit(1)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output.csv')
    if args.save_viz:
        os.makedirs(args.viz_dir, exist_ok=True)

    images = load_csv(args.data)
    t0 = time.time()

    n_workers = min(multiprocessing.cpu_count() or 1, 12)
    args_list = [(item['path'], item['image_id']) for item in images]

    decode_flag = str(args.decode).lower() not in ('no', 'false', '0')
    no_decode = not decode_flag
    mode_msg = "detection + decoding" if decode_flag else "detection only (decoding disabled)"
    print(f"Processing {len(images)} images with {n_workers} worker(s) - mode: {mode_msg}")
    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(no_decode,)) as pool:
        results = list(pool.map(_process_image_worker, args_list, chunksize=8))

    total_qr = sum(r['count'] for r in results)
    elapsed = time.time() - t0

    print("-" * 55)
    print(f"Total QR found : {total_qr}")
    print(f"Total time     : {elapsed:.2f}s  ({elapsed/max(len(images),1)*1000:.0f}ms/img)")
    print(f"Output         : {output_path}")
    write_csv(results, output_path)


if __name__ == '__main__':
    main()
