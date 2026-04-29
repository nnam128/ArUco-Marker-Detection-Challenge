import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import generate_augmented_images


MODEL_PATH = "models/aruco_classifier_model_epoch50.keras"
PATCH_SIZE = 64
WARP_SIZE = 64
CNN_THRESHOLD = 0.5
PADDING = 10


def load_cnn_model(model_path=MODEL_PATH):
    model = load_model(model_path)
    print(f"Loaded CNN model from: {model_path}")
    return model



def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)



def warp_candidate(image, corners, output_size=WARP_SIZE):
    src = order_corners(corners)

    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))

    return warped



def prepare_patch_for_cnn(image, corners):
    warped = warp_candidate(image, corners, output_size=WARP_SIZE)
    patch = cv2.resize(warped, (PATCH_SIZE, PATCH_SIZE))
    patch = patch.astype(np.float32) / 255.0
    patch = np.expand_dims(patch, axis=0)
    return patch

from src.preprocess import generate_augmented_images



def build_augmented_context(image):
    aug_dict = generate_augmented_images(image)
    return aug_dict


def get_cnn_input_image(cand, aug_dict, fallback_image):

    source_name = None

    if isinstance(cand, dict):
        source_name = cand.get("source", None)

    if source_name is not None and source_name in aug_dict:
        return aug_dict[source_name]["image"]
    
    # fallback (an toàn)
    return fallback_image



def is_real_marker_consistent(model, image, cand, threshold=0.5):
    """
    CNN prediction nhưng đảm bảo input giống training distribution
    """

    aug_dict = build_augmented_context(image)

    target_img = get_cnn_input_image(
        cand,
        aug_dict,
        fallback_image=image
    )

    return is_real_marker(
        model,
        target_img,
        cand["corners"],
        threshold=threshold
    )


def is_real_marker(model, image, corners, threshold=CNN_THRESHOLD):
    patch = prepare_patch_for_cnn(image, corners)
    prob = float(model.predict(patch, verbose=0)[0][0])

    # class 1 = real
    return prob >= threshold, prob


def create_roi_with_padding(image, corners, padding=PADDING):
    h, w = image.shape[:2]
    pts = order_corners(corners)

    x_min = max(int(np.min(pts[:, 0])) - padding, 0)
    y_min = max(int(np.min(pts[:, 1])) - padding, 0)
    x_max = min(int(np.max(pts[:, 0])) + padding, w)
    y_max = min(int(np.max(pts[:, 1])) + padding, h)

    roi = image[y_min:y_max, x_min:x_max].copy()

    return roi, x_min, y_min


# =========================================================
# STRICT ARUCO DETECTION
# =========================================================
def strict_detect_aruco(roi):
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    params = cv2.aruco.DetectorParameters()

    # stricter settings
    params.minMarkerPerimeterRate = 0.05
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 5
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(roi)

    return corners, ids


def postprocess_candidates(image, candidates, model):
    results = []

    for candidate in candidates:
        is_real, prob = is_real_marker(model, image, candidate)

        if not is_real:
            continue

        roi, offset_x, offset_y = create_roi_with_padding(
            image,
            candidate,
            padding=PADDING
        )

        corners, ids = strict_detect_aruco(roi)

        if ids is None:
            continue

        for marker_corners, marker_id in zip(corners, ids):
            pts = marker_corners[0]
            pts = order_corners(pts)

            top_left = pts[0]

            x = int(top_left[0] + offset_x)
            y = int(top_left[1] + offset_y)
            marker_id = int(marker_id[0])

            results.append((marker_id, x, y))

    return results



def format_output(results):
    output = []

    for marker_id, x, y in results:
        output.extend([str(marker_id), str(x), str(y)])

    return " ".join(output)


if __name__ == "__main__":
    print("postprocess.py ready")
    print("Use postprocess_candidates(image, candidates, model)")
