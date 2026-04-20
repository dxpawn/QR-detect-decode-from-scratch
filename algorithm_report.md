# Algorithm Report: QR Code Detection and Decoding System

## 1. Overview

This system detects, localizes (4-corner bounding quadrilateral), and optionally decodes QR codes in images. It is built entirely from scratch using only **OpenCV** (for image I/O and basic operations), **NumPy** (for numerical computation), and custom implementations of all QR-specific algorithms.

**No high-level QR libraries** (such as `pyzbar`, `cv2.QRCodeDetector`, or `zxing`) are used.

## 2. Detection Pipeline

### 2.1 Image Preprocessing

The system uses **6 parallel preprocessing strategies** to handle varying image conditions (brightness, contrast, noise, blur):

| # | Strategy | Purpose |
|---|----------|---------|
| 1 | CLAHE (clip=3.0) + Gaussian Adaptive Threshold (block=11) | General-purpose: good for medium contrast |
| 2 | CLAHE + Otsu's Threshold | Best for bimodal intensity distributions |
| 3 | Unsharp Masking (1.8x) + Adaptive Threshold (block=15) | Enhances blurred or low-contrast edges |
| 4 | Raw Grayscale + Adaptive Threshold (block=7) | Small QR codes with fine detail |
| 5 | CLAHE + Mean Adaptive Threshold (block=21) | Smooth lighting gradient compensation |
| 6 | CLAHE + Adaptive Threshold (block=25) + Morphological Close | Handles broken contours from noise |

Each strategy produces a binary image, and candidates from all strategies are merged for robust detection.

### 2.2 Finder Pattern Detection

QR codes have three **finder patterns** (position detection patterns) — concentric squares with a 1:1:3:1:1 black-white-black-white-black ratio.

**Detection Method: Contour Hierarchy Analysis**

1. Run `cv2.findContours` with `RETR_TREE` mode to obtain full contour hierarchy
2. Search for **3-level nested contours** (outer → middle → inner) with area ratios consistent with the 1:1:3:1:1 pattern:
   - `outer/middle ≈ 49/25 ≈ 1.96` (allowed range: 1.2–12×)
   - `middle/inner ≈ 25/9 ≈ 2.78` (allowed range: 1.2–12×)
3. Filter by **aspect ratio** (max/min dimension ratio ≤ 2.5)

**Verification: Scan-Line Ratio Analysis**

For each candidate, scan along 4 directions (horizontal, vertical, diagonal, anti-diagonal) through the center. Count run-length transitions and verify the 1:1:3:1:1 ratio with tolerance:

$$|r_i - e_i \cdot u| \leq 0.85 \cdot u + 1.0$$

where $u = \text{total} / 7$ is the unit module size, $r_i$ is the observed run length, and $e_i \in \{1, 1, 3, 1, 1\}$ is the expected multiplier.

A candidate passes if at least 2 of 4 scan directions match.

### 2.3 Cross-Strategy Candidate Merging

Candidates from different strategies that are spatially close (within 1.2× the finder pattern side length) are merged, keeping the largest. This eliminates duplicates while ensuring coverage across strategies.

### 2.4 Triplet Grouping

Three finder patterns form a valid QR code if they satisfy:

1. **Size consistency**: Pairwise finder-pattern-side-length ratio ≤ 1.85
2. **Right-angle geometry**: The largest angle in the triangle ≈ 90° (range: 50°–130°)
3. **Side ratio**: The two sides from the right-angle vertex have ratio ≤ 2.0
4. **Hypotenuse ratio**: Hypotenuse / average side ∈ (1.0, 2.0), ideally √2 ≈ 1.414
5. **Timing pattern check**: Sampling pixels along the line between two finder centers to verify alternating dark/light modules
6. **Version estimate**: The computed QR version must be in [1, 40]
7. **Collinearity check**: The two arms from the corner must not be too parallel (|cos θ| ≤ 0.65)

**Scoring**: Valid triplets are ranked by: `2.5 × |angle − 90°| + 100 × (side_ratio − 1) + 100 × |hyp_ratio − √2| + 60 × size_variance`

Best-scoring triplets are greedily assigned, ensuring each finder pattern is used at most once.

### 2.5 Corner Estimation

Given the three finder pattern centers (TL, TR, BL), the fourth corner (BR) is estimated using:

1. Determine TR vs BL using cross-product orientation
2. Compute module size from finder pattern dimensions
3. Extend each corner outward by 3.5 modules
4. Estimate BR as a weighted average of parallelogram projection (60%) and extension projection (40%)

## 3. QR Code Decoding (Optional)

### 3.1 Perspective Rectification

Using `cv2.getPerspectiveTransform` and `cv2.warpPerspective`, the detected QR region is warped to a rectified NxN grid, where N = 4V + 17 and V is the QR version.

### 3.2 Bit Matrix Sampling

Each module is sampled at its center (3×3 pixel area average for robustness). An **iterative mean threshold** is applied to binarize:

$$T_{k+1} = \frac{\text{mean}(pixels < T_k) + \text{mean}(pixels \geq T_k)}{2}$$

Iteration continues until convergence (|ΔT| < 1.0).

### 3.3 Format Information

The 15-bit format info is read from two locations (horizontal and vertical strips around the top-left finder). After XOR with mask `0x5412`, BCH(15,5) error correction finds the closest valid format, extracting the error correction level (L/M/Q/H) and mask pattern (0–7).

### 3.4 Data Extraction

The zigzag traversal reads data modules in 2-column pairs, right-to-left, alternating upward/downward. Function modules (finders, alignment patterns, timing, format/version info, dark module) are skipped.

### 3.5 Error Correction

**Reed-Solomon** error correction over GF(2⁸) uses:
- **Berlekamp-Massey** algorithm for error locator polynomial
- **Forney algorithm** for error magnitude
- Generator polynomial: `x⁸ + x⁴ + x³ + x² + 1` (primitive polynomial `0x11D`)

The implementation supports all 40 QR versions across all 4 EC levels.

### 3.6 Data Decoding

The decoded data stream supports 4 encoding modes:
- **Numeric** (0001): Groups of 3→10 bits, 2→7 bits, 1→4 bits
- **Alphanumeric** (0010): Pairs→11 bits, single→6 bits
- **Byte** (0100): UTF-8/Latin-1, 8 bits per character
- **Kanji** (1000): Shift-JIS, 13 bits per character

Mixed-mode segments are handled sequentially.

## 4. Performance

| Metric | Validation Set | Training Set |
|--------|---------------|-------------|
| Precision | 0.627 | 0.651 |
| Recall | 0.443 | 0.382 |
| F1 Score | 0.519 | 0.481 |
| Avg. Latency | ~0.21s/image | ~0.22s/image |
| Total Images | 309 | 1083 |

## 5. Design Decisions

1. **No 2-level nesting fallback**: Tested but produced too many false positives (F1 dropped from 0.52 to 0.32)
2. **6 strategies, not 10+**: Extended strategies added ~100s of latency without improving recall
3. **Timing pattern verification**: Critical for filtering "ghost" QR codes — false triplets from non-QR geometric structures
4. **Cross-strategy merging**: Enables finding QR codes whose 3 finder patterns are each detected by different strategies
