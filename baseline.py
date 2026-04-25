import os
import cv2
import pandas as pd
from tqdm import tqdm

TEST_DIR = "data/aruco_data/train"
OUTPUT_CSV = "submission.csv"

def get_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_ARUCO_MIP_36h12
    )

    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters
    )

    return detector

def detect_markers(image, detector):
    corners, ids, _ = detector.detectMarkers(image)

    predictions = []

    if ids is None:
        return predictions

    for marker_corners, marker_id in zip(corners, ids):
        marker_id = int(marker_id[0])

        top_left = marker_corners[0][0]

        x = float(top_left[0])
        y = float(top_left[1])

        predictions.append((marker_id, x, y))

    return predictions


def format_prediction(predictions):
    if not predictions:
        return ""

    result = []

    for marker_id, x, y in predictions:
        result.append(
            f"{marker_id} {x:.3f} {y:.3f}"
        )

    return " ".join(result)


def main():
    detector = get_detector()

    image_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])

    rows = []

    for image_file in tqdm(image_files):
        image_path = os.path.join(TEST_DIR, image_file)

        image = cv2.imread(image_path)

        if image is None:
            continue

        predictions = detect_markers(image, detector)

        prediction_string = format_prediction(predictions)

        image_id = os.path.splitext(image_file)[0]

        rows.append({
            "image_id": image_id,
            "prediction_string": prediction_string
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()