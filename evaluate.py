import pandas as pd
import numpy as np
import math


# CONFIG
SIGMA = 0.02
LAMBDA = 0.5
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1200


# PARSE PREDICTION STRING
def parse_prediction_string(pred_str):
    if pd.isna(pred_str) or str(pred_str).strip() == "":
        return []

    tokens = str(pred_str).strip().split()

    if len(tokens) % 3 != 0:
        print(f"WARNING: invalid prediction_string -> {pred_str}")
        return []

    markers = []

    for i in range(0, len(tokens), 3):
        marker_id = int(float(tokens[i]))
        x = float(tokens[i + 1])
        y = float(tokens[i + 2])

        markers.append({
            "id": marker_id,
            "x": x,
            "y": y
        })

    return markers


# DISTANCE SCORE
def compute_distance_score(pred_x, pred_y, gt_x, gt_y,
                           image_w=IMAGE_WIDTH,
                           image_h=IMAGE_HEIGHT,
                           sigma=SIGMA):

    d = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    diag = math.sqrt(image_w ** 2 + image_h ** 2)

    d_norm = d / diag

    return math.exp(- (d_norm ** 2) / (2 * sigma ** 2))


# IMAGE SCORE
def compute_image_score(gt_markers,
                        pred_markers,
                        sigma=SIGMA,
                        spam_lambda=LAMBDA,
                        image_w=IMAGE_WIDTH,
                        image_h=IMAGE_HEIGHT):

    N_gt = len(gt_markers)

    if N_gt == 0:
        return 0.0

    used_pred = set()
    sum_metric = 0.0
    matched_count = 0

    for gt in gt_markers:
        gt_id = gt["id"]

        best_score = -1
        best_pred_idx = None

        for i, pred in enumerate(pred_markers):
            if i in used_pred:
                continue
            if pred["id"] != gt_id:
                continue

            score = compute_distance_score(
                pred["x"], pred["y"],
                gt["x"], gt["y"],
                image_w=image_w,
                image_h=image_h,
                sigma=sigma
            )

            if score > best_score:
                best_score = score
                best_pred_idx = i

        if best_pred_idx is not None:
            used_pred.add(best_pred_idx)
            sum_metric += best_score
            matched_count += 1

    extra_predictions = len(pred_markers) - matched_count

    denominator = N_gt + spam_lambda * extra_predictions

    return sum_metric / denominator if denominator > 0 else 0.0


# EVALUATION
def evaluate_csv(
    gt_csv_path,
    pred_csv_path,
    sigma=SIGMA,
    spam_lambda=LAMBDA,
    image_w=IMAGE_WIDTH,
    image_h=IMAGE_HEIGHT
):

    gt_df = pd.read_csv(gt_csv_path)
    pred_df = pd.read_csv(pred_csv_path)

    gt_map = dict(zip(gt_df["image_id"], gt_df["prediction_string"]))
    pred_map = dict(zip(pred_df["image_id"], pred_df["prediction_string"]))

    all_image_ids = sorted(set(gt_map.keys()))

    scores = []
    stats = []

    print("=" * 80)
    print("PER IMAGE SCORE + ERROR ANALYSIS")
    print("=" * 80)

    for image_id in all_image_ids:

        gt_markers = parse_prediction_string(gt_map.get(image_id, ""))
        pred_markers = parse_prediction_string(pred_map.get(image_id, ""))

        score = compute_image_score(
            gt_markers,
            pred_markers,
            sigma=sigma,
            spam_lambda=spam_lambda,
            image_w=image_w,
            image_h=image_h
        )

        matched_gt = set()
        matched_pred = set()

        for gt in gt_markers:
            gt_id = gt["id"]

            best_idx = None
            best_score = -1

            for i, pred in enumerate(pred_markers):
                if i in matched_pred:
                    continue
                if pred["id"] != gt_id:
                    continue

                d = compute_distance_score(
                    pred["x"], pred["y"],
                    gt["x"], gt["y"],
                    image_w=image_w,
                    image_h=image_h,
                    sigma=sigma
                )

                if d > best_score:
                    best_score = d
                    best_idx = i

            if best_idx is not None:
                matched_gt.add(gt_id)
                matched_pred.add(best_idx)

        missing = len(gt_markers) - len(matched_gt)
        extra = len(pred_markers) - len(matched_pred)

        stats.append((image_id, missing, extra))

        print(f"{image_id}: score={score:.6f} | missing={missing} | extra={extra}")

        scores.append(score)

    final_score = np.mean(scores)

    total_missing = sum(s[1] for s in stats)
    total_extra = sum(s[2] for s in stats)

    print("\n" + "=" * 80)
    print("GLOBAL RESULT")
    print("=" * 80)

    print(f"FINAL SCORE: {final_score:.6f}")
    print(f"TOTAL MISSING MARKERS: {total_missing}")
    print(f"TOTAL EXTRA MARKERS: {total_extra}")
    print(f"AVG MISSING / IMAGE: {total_missing / len(stats):.3f}")
    print(f"AVG EXTRA / IMAGE: {total_extra / len(stats):.3f}")

    return final_score


# MAIN
if __name__ == "__main__":
    GT_CSV = "test.csv"
    PRED_CSV = "submission.csv"

    evaluate_csv(GT_CSV, PRED_CSV)