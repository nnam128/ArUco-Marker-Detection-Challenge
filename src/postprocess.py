import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import generate_augmented_images


MODEL_PATH = "models/aruco_classifier_model_epoch50.keras"
PATCH_SIZE = 64
WARP_SIZE = 64
CNN_THRESHOLD = 0.5
PADDING = 10
PADDING_RATIO = 0.3


def load_cnn_model(model_path=MODEL_PATH):
    model = load_model(model_path)
    print(f"Loaded CNN model from: {model_path}")
    return model



def order_corners(pts):
    pts = normalize_corners(pts)

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

    return is_real_marker(model, image, cand["corners"], threshold=threshold)

def normalize_corners(corners):
    corners = np.array(corners)

    # case OpenCV: (1,4,2)
    if corners.shape == (1, 4, 2):
        corners = corners[0]

    # case weird nested list
    corners = np.squeeze(corners)

    if corners.shape != (4, 2):
        raise ValueError(f"Invalid corners shape: {corners.shape}")

    return corners.astype(np.float32)

def is_real_marker(model, image, corners, threshold=CNN_THRESHOLD):
    patch = prepare_patch_for_cnn(image, corners)
    prob = float(model.predict(patch, verbose=0)[0][0])

    # class 1 = real
    return prob >= threshold, prob


def create_roi_with_padding(image, corners, padding=PADDING):
    h, w = image.shape[:2]
    pts = normalize_corners(corners)
    
    marker_w = np.max(pts[:, 0]) - np.min(pts[:, 0])
    marker_h = np.max(pts[:, 1]) - np.min(pts[:, 1])
    
    padding = max(PADDING, int(min(marker_w, marker_h) * PADDING_RATIO))

    # floor/ceil để cover toàn bộ vùng sub-pixel
    x_min = max(int(np.floor(np.min(pts[:, 0]) - padding)), 0)
    y_min = max(int(np.floor(np.min(pts[:, 1]) - padding)), 0)
    x_max = min(int(np.ceil(np.max(pts[:, 0]) + padding)), w)
    y_max = min(int(np.ceil(np.max(pts[:, 1]) + padding)), h)
    
    
    roi = image[y_min:y_max, x_min:x_max].copy()
    
    return roi, x_min, y_min

# STRICT ARUCO DETECTION
def strict_detect_aruco(roi, refine_method=cv2.aruco.CORNER_REFINE_SUBPIX, for_roi=False):
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_ARUCO_MIP_36H12
    )

    params = cv2.aruco.DetectorParameters()

    # refine góc
    params.cornerRefinementMethod = refine_method
    

    # adaptive threshold
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 31
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 5

    # lọc candidate chặt hơn
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 0

    # decode chặt hơn
    params.minOtsuStdDev = 5.0
    params.errorCorrectionRate = 0.01


    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(roi)
    
    return corners, ids

def strict_detect_aruco_roi(roi, refine_method=cv2.aruco.CORNER_REFINE_SUBPIX, for_roi=False):
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_ARUCO_MIP_36H12
    )
    
    h, w = roi.shape[:2]
    side = min(h, w)

    params = cv2.aruco.DetectorParameters()

    # refine góc
    params.cornerRefinementMethod = refine_method
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.03
    params.relativeCornerRefinmentWinSize = 0.4

    # adaptive threshold
    win_min = max(3, int(side * 0.10)); win_min += (win_min % 2 == 0)
    win_max = max(win_min + 2, int(side * 0.9)); win_max += (win_max % 2 == 0)
    step    = max(2, (win_max - win_min) // 8)
    params.adaptiveThreshWinSizeMin  = win_min
    params.adaptiveThreshWinSizeMax  = win_max
    params.adaptiveThreshWinSizeStep = step
    params.adaptiveThreshConstant    = 7

    # lọc candidate chặt hơn
    params.minMarkerPerimeterRate = 0.3
    params.maxMarkerPerimeterRate = 2.2
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.03
    params.minDistanceToBorder = 0
    
    params.perspectiveRemovePixelPerCell = max(4, int(side / 15))
    params.perspectiveRemoveIgnoredMarginPerCell = 0.15

    params.minSideLengthCanonicalImg = 16


    params.markerBorderBits = 1
    params.minOtsuStdDev = 0

    params.errorCorrectionRate = 0.3

    params.detectInvertedMarker = True

    params.useAruco3Detection = True
    params.maxErroneousBitsInBorderRate = 0.25


    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(roi)
    
    return corners, ids


def postprocess_candidates(image, candidates, model):
    results = []

    for candidate in candidates:
        is_real, prob = is_real_marker_consistent(model, image, candidate)

        if not is_real:
            continue

        roi, offset_x, offset_y = create_roi_with_padding(
            image,
            candidate["corners"],
            padding=PADDING
        )
        

        corners, ids = strict_detect_aruco_roi(roi)

        if ids is None:
            continue

        for marker_corners, marker_id in zip(corners, ids):
            pts = marker_corners[0]
            
            top_left = pts[0]

            # GIỮ NGUYÊN FLOAT - Không ép về int()
            x = float(top_left[0] + offset_x)
            y = float(top_left[1] + offset_y)
            marker_id = int(marker_id[0])

            results.append((marker_id, x, y))

    return results

def format_output(results):
    output = []
    results_sorted = sorted(results, key=lambda item: item[0])
    
    for marker_id, x, y in results_sorted:
        output.extend([str(marker_id), f"{x:.3f}", f"{y:.3f}"])

    return " ".join(output)


if __name__ == "__main__":
    print("postprocess.py ready")
    print("Use postprocess_candidates(image, candidates, model)")
