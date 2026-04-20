"""
F1 Score Evaluator for QR Code Detection
Computes Precision, Recall, F1 using IoU-based greedy matching.
Uses Sutherland-Hodgman polygon clipping for IoU on quadrilaterals.
"""
import csv
import sys
import numpy as np
from collections import defaultdict


def polygon_area(pts):
    """Shoelace formula for polygon area."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def line_intersect(p1, p2, p3, p4):
    """Find intersection point of two line segments."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def sutherland_hodgman(subject, clip):
    """Sutherland-Hodgman polygon clipping algorithm."""
    output = list(subject)
    if len(output) == 0:
        return []

    for i in range(len(clip)):
        if len(output) == 0:
            return []
        input_list = output
        output = []
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]

        for j in range(len(input_list)):
            current = input_list[j]
            prev = input_list[j - 1]

            # Check if points are inside (left of edge)
            def is_inside(p):
                return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - \
                       (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]) >= 0

            curr_inside = is_inside(current)
            prev_inside = is_inside(prev)

            if curr_inside:
                if not prev_inside:
                    intersection = line_intersect(prev, current, edge_start, edge_end)
                    if intersection:
                        output.append(intersection)
                output.append(current)
            elif prev_inside:
                intersection = line_intersect(prev, current, edge_start, edge_end)
                if intersection:
                    output.append(intersection)

    return output


def compute_iou(quad1, quad2):
    """Compute IoU between two quadrilaterals."""
    # Ensure clockwise ordering
    poly1 = [(quad1[i][0], quad1[i][1]) for i in range(4)]
    poly2 = [(quad2[i][0], quad2[i][1]) for i in range(4)]

    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)

    if area1 < 1 or area2 < 1:
        return 0.0

    intersection = sutherland_hodgman(poly1, poly2)
    if len(intersection) < 3:
        return 0.0

    inter_area = polygon_area(intersection)
    union_area = area1 + area2 - inter_area

    if union_area < 1e-10:
        return 0.0

    return inter_area / union_area


def load_output_csv(path):
    """Load QR detection output CSV."""
    data = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 10:
                continue
            img_id = row[0].strip()
            qr_idx = row[1].strip()
            if qr_idx == '' or qr_idx == '-1':
                if img_id not in data:
                    data[img_id] = []
                continue
            try:
                coords = [float(row[i]) for i in range(2, 10)]
                corners = [(coords[0], coords[1]), (coords[2], coords[3]),
                          (coords[4], coords[5]), (coords[6], coords[7])]
                content = row[10].strip() if len(row) > 10 else ''
                data[img_id].append({'corners': corners, 'content': content})
            except (ValueError, IndexError):
                continue
    return data


def evaluate(gt_path, pred_path, iou_threshold=0.5):
    """Evaluate predictions against ground truth."""
    gt = load_output_csv(gt_path)
    pred = load_output_csv(pred_path)

    tp = 0
    fp = 0
    fn = 0
    content_correct = 0

    all_images = set(list(gt.keys()) + list(pred.keys()))

    for img_id in all_images:
        gt_boxes = gt.get(img_id, [])
        pred_boxes = pred.get(img_id, [])

        matched_gt = set()

        for p_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p_box['corners'], g_box['corners'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                # Check content accuracy
                if p_box.get('content', ''):
                    gt_content = gt_boxes[best_gt_idx].get('content', '')
                    if p_box['content'].strip().lower() == gt_content.strip().lower():
                        content_correct += 1
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    content_acc = content_correct / tp if tp > 0 else 0.0

    print(f"{'='*55}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    if tp > 0:
        print(f"  Content Accuracy: {content_acc:.4f} ({content_correct}/{tp})")
    print(f"{'='*55}")

    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision,
            'recall': recall, 'f1': f1, 'content_acc': content_acc}


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <ground_truth.csv> <predictions.csv>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
