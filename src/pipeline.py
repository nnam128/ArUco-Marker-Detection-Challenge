import cv2
import numpy as np
from src.preprocess import generate_augmented_images_strict,generate_augmented_images, aug_clahe_gamma_sharp, aug_gaussian_noise, aug_dilation, aug_erosion, aug_motion_blur
from src.detector import extract_candidates, filter_and_cluster_candidates, map_corners_to_original
from src.postprocess import (
    load_cnn_model, 
    is_real_marker_consistent, 
    create_roi_with_padding, 
    strict_detect_aruco, 
    strict_detect_aruco_roi,
)
DUPLICATE_DISTANCE_THRESHOLD = 10

def detect_id_in_roi_variants(roi_bgr):
    h, w = roi_bgr.shape[:2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    roi_2x  = cv2.resize(roi_bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    roi_3x  = cv2.resize(roi_bgr, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    gray_2x = cv2.cvtColor(roi_2x, cv2.COLOR_BGR2GRAY)
    gray_3x = cv2.cvtColor(roi_3x, cv2.COLOR_BGR2GRAY)

    # --- Helper ---
    def to_bgr(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def gamma(g, val):
        return np.uint8(np.clip(
            np.power(g.astype(np.float32) / 255.0, 1.0 / val) * 255, 0, 255
        ))

    def clahe(g, clip, grid):
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(g)

    gray_norm    = cv2.normalize(gray,    None, 0, 255, cv2.NORM_MINMAX)
    gray_2x_norm = cv2.normalize(gray_2x, None, 0, 255, cv2.NORM_MINMAX)
    gray_3x_norm = cv2.normalize(gray_3x, None, 0, 255, cv2.NORM_MINMAX)

    def adapt_gauss(g, win):
        return to_bgr(cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, win, 2
        ))

    def adapt_mean(g, win):
        return to_bgr(cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, win, 2
        ))

    variants = [

        # =====================================================
        # NHÓM 1: UPSCALE — base cho tất cả nhóm sau
        # =====================================================
        (roi_bgr, 1.0),
        (roi_2x,  2.0),
        (roi_3x,  3.0),

        # =====================================================
        # NHÓM 2: DENOISE — giảm noise trước khi ArUco threshold
        # =====================================================
        (cv2.medianBlur(roi_2x, 3),              2.0),
        (cv2.medianBlur(roi_2x, 5),              2.0),
        (cv2.bilateralFilter(roi_bgr, 5, 40, 40), 1.0),
        (cv2.bilateralFilter(roi_2x,  7, 60, 60), 2.0),

        # =====================================================
        # NHÓM 3: CONTRAST ENHANCEMENT (BGR pipeline)
        # Tăng sáng tuyến tính — khác nhau về alpha/beta
        # =====================================================
        (cv2.convertScaleAbs(roi_bgr, alpha=2.0, beta=30),  1.0),
        (cv2.convertScaleAbs(roi_bgr, alpha=3.0, beta=80),  1.0),
        (cv2.convertScaleAbs(roi_2x,  alpha=2.0, beta=30),  2.0),
        (cv2.convertScaleAbs(roi_2x,  alpha=3.0, beta=80),  2.0),
        # Giảm sáng — overexposed
        (cv2.convertScaleAbs(roi_bgr, alpha=0.4, beta=-30), 1.0),
        (cv2.convertScaleAbs(roi_2x,  alpha=0.4, beta=-30), 2.0),

        # =====================================================
        # NHÓM 4: GRAY ENHANCEMENT (Gray pipeline)
        # Mỗi technique 1 lần trên gray, 1 lần trên gray_2x
        # Không lặp lại ở nhóm sau
        # =====================================================

        # 4a: Gamma — phi tuyến, tốt cho shadow
        (to_bgr(gamma(gray,    2.0)), 1.0),
        (to_bgr(gamma(gray,    3.0)), 1.0),
        (to_bgr(gamma(gray_2x, 2.0)), 2.0),
        # Gamma < 1 cho overexposed
        (to_bgr(gamma(gray_2x, 0.5)), 2.0),

        # 4b: CLAHE — contrast cục bộ
        (to_bgr(clahe(gray,    4.0, 8)), 1.0),
        (to_bgr(clahe(gray,    8.0, 4)), 1.0),
        (to_bgr(clahe(gray_2x, 4.0, 8)), 2.0),
        (to_bgr(clahe(gray_2x, 8.0, 4)), 2.0),

        # 4c: equalizeHist — global, nhanh
        (to_bgr(cv2.equalizeHist(gray)),    1.0),
        (to_bgr(cv2.equalizeHist(gray_2x)), 2.0),

        # 4d: Sharpen
        (aug_clahe_gamma_sharp(roi_bgr), 1.0),
        (aug_clahe_gamma_sharp(roi_2x),  2.0),
        (cv2.addWeighted(roi_2x, 1.5, cv2.GaussianBlur(roi_2x, (0,0), 1.0), -0.5, 0), 2.0),
        (cv2.addWeighted(roi_2x, 2.0, cv2.GaussianBlur(roi_2x, (0,0), 2.0), -1.0, 0), 2.0),

        # =====================================================
        # NHÓM 5: ADAPTIVE THRESHOLD
        # 3 window size chuẩn × 3 input source
        # Source: gray | clahe(gray) | median(gray)
        # Không lặp source
        # =====================================================

        # 5a: raw gray → adaptive (3 windows)
        (adapt_gauss(gray, 7),  1.0),
        (adapt_gauss(gray, 11), 1.0),
        (adapt_gauss(gray, 21), 1.0),
        (adapt_mean(gray,  11), 1.0),

        # 5b: clahe(gray) → adaptive — tăng contrast trước
        (adapt_gauss(clahe(gray, 4.0, 8), 11), 1.0),
        (adapt_gauss(clahe(gray, 8.0, 4), 11), 1.0),
        (adapt_mean(clahe(gray,  4.0, 8), 11), 1.0),

        # 5c: median(gray) → adaptive — denoise trước
        (adapt_gauss(cv2.medianBlur(gray, 3), 11), 1.0),
        (adapt_mean(cv2.medianBlur(gray,  3), 11), 1.0),

        # 5d: upscale → adaptive (window tương ứng 2x)
        (adapt_gauss(gray_2x, 11), 2.0),
        (adapt_gauss(gray_2x, 21), 2.0),
        (adapt_mean(gray_2x,  11), 2.0),

        # =====================================================
        # NHÓM 6: MORPHOLOGY
        # =====================================================
        (aug_dilation(roi_bgr), 1.0),
        (aug_erosion(roi_bgr),  1.0),
        (aug_dilation(roi_2x),  2.0),
        (aug_erosion(roi_2x),   2.0),

        # =====================================================
        # NHÓM 7: INVERT — marker sáng trên nền tối
        # Đầy đủ branch: raw / denoise / adaptive / normalize
        # =====================================================
        (cv2.bitwise_not(roi_bgr), 1.0),
        (cv2.bitwise_not(roi_2x),  2.0),
        (cv2.bitwise_not(roi_3x),  3.0),

        # invert + tăng sáng (sau invert marker vẫn tối)
        (cv2.convertScaleAbs(cv2.bitwise_not(roi_bgr), alpha=2.0, beta=30), 1.0),
        (cv2.convertScaleAbs(cv2.bitwise_not(roi_2x),  alpha=2.0, beta=30), 2.0),

        # invert + adaptive — tách edge sau invert
        (adapt_gauss(cv2.cvtColor(cv2.bitwise_not(roi_bgr), cv2.COLOR_BGR2GRAY), 11), 1.0),
        (adapt_gauss(cv2.cvtColor(cv2.bitwise_not(roi_2x),  cv2.COLOR_BGR2GRAY), 11), 2.0),
        (adapt_gauss(cv2.cvtColor(cv2.bitwise_not(roi_2x),  cv2.COLOR_BGR2GRAY), 21), 2.0),

        # invert + normalize + adaptive — case siêu tối sau invert
        (adapt_gauss(cv2.normalize(
            cv2.cvtColor(cv2.bitwise_not(roi_2x), cv2.COLOR_BGR2GRAY),
            None, 0, 255, cv2.NORM_MINMAX
        ), 11), 2.0),
        

        # =====================================================åå
        # NHÓM 8: SIÊU TỐI — NORMALIZE (augment thật sự mới)å
        # Normalize stretch full range 0-255 TRƯỚC KHI threshold
        # Khác hoàn toàn CLAHE/gamma (đó là contrast local/gamma)
        # =====================================================

        # 8a: normalize raw
        (to_bgr(gray_norm),    1.0),
        (to_bgr(gray_2x_norm), 2.0),
        (to_bgr(gray_3x_norm), 3.0),

        # 8b: normalize + adaptive — KHÔNG lặp source từ nhóm 5
        (adapt_gauss(gray_norm,     7),  1.0),
        (adapt_gauss(gray_norm,    11),  1.0),
        (adapt_gauss(gray_2x_norm, 11),  2.0),
        (adapt_gauss(gray_2x_norm, 21),  2.0),
        (adapt_gauss(gray_3x_norm, 11),  3.0),
        (adapt_mean(gray_norm,     11),  1.0),
        (adapt_mean(gray_2x_norm,  11),  2.0),

        # 8c: normalize + clahe + adaptive
        # KHÁC nhóm 5b (đó là clahe(gray), đây là clahe(gray_norm))
        (adapt_gauss(clahe(gray_norm,    8.0, 4), 11), 1.0),
        (adapt_gauss(clahe(gray_2x_norm, 8.0, 4), 11), 2.0),
        (adapt_gauss(clahe(gray_3x_norm, 8.0, 4), 11), 3.0),

        # 8d: normalize + median + adaptive
        (adapt_gauss(cv2.medianBlur(gray_2x_norm, 3), 11), 2.0),
        (adapt_gauss(cv2.medianBlur(gray_3x_norm, 3), 11), 3.0),

        # 8e: normalize + gamma + adaptive
        # gamma sau normalize → tăng thêm vùng tối đã stretch
        (adapt_gauss(gamma(gray_2x_norm, 2.0), 11), 2.0),
        (adapt_gauss(gamma(gray_3x_norm, 2.0), 11), 3.0),
    ]
    for variant, scale_factor in variants:
        if len(variant.shape) == 2:
            variant = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)

        corners, ids = strict_detect_aruco_roi(variant)
        if ids is not None:
            if scale_factor != 1.0:
                corners = [c / scale_factor for c in corners]
            return corners, ids

    return None, None


def run_aruco_pipeline(image_path, model):
    image = cv2.imread(image_path)
    if image is None: return "Error: Image not found"


    final_results = {}

    corners_orig, ids_orig = strict_detect_aruco(image)
    if ids_orig is not None:
        for c, mid in zip(corners_orig, ids_orig):
            m_id = int(mid[0])
            pts = c[0] 
            if m_id not in final_results:
                    final_results[m_id] = []
            #print(f"Found ID {m_id} in original image at position ({pts[0][0]:.2f}, {pts[0][1]:.2f})")
            final_results[m_id].append((float(pts[0][0]), float(pts[0][1])))

    aug_dict = generate_augmented_images_strict(image)
    for name, data in aug_dict.items():
        if name == "original": continue
        
        corners, ids = strict_detect_aruco(data["image"])
        if ids is not None:
            for c, mid in zip(corners, ids):
                
                m_id = int(mid[0])
                orig_corners = map_corners_to_original(c[0], data["transform_info"])
                x, y = float(orig_corners[0][0]), float(orig_corners[0][1])
                
                if m_id not in final_results:
                    final_results[m_id] = []
                
                
                should_add = True

                for old_x, old_y in final_results[m_id]:
                    dist = ((x - old_x) ** 2 + (y - old_y) ** 2) ** 0.5

                    if dist < DUPLICATE_DISTANCE_THRESHOLD:
                        should_add = False
                        break

                if should_add:
                    #print(f"Found ID {m_id} in augmentation '{name}' at mapped position ({x:.2f}, {y:.2f})")
                    final_results[m_id].append((x, y))

    aug_dict = generate_augmented_images(image)
    raw_candidates = extract_candidates(image, aug_dict)
    # Tăng min_votes nếu muốn bớt nhiễu, dist_thresh nên khớp với size marker dự kiến
    final_candidates = filter_and_cluster_candidates(raw_candidates)

    for cand in final_candidates:
        cand_tl = cand["corners"][0] 
        
        already_exists = False

        for positions in final_results.values():
            for pos in positions:
                if np.linalg.norm(cand_tl - np.array(pos)) < 40:
                    already_exists = True
                    break
            if already_exists:
                break

        if already_exists:
            continue
        
        is_real, prob = is_real_marker_consistent(model, image, cand, threshold=0.1)

        if is_real:
            roi, offset_x, offset_y = create_roi_with_padding(image, cand["corners"], padding=10)
            corners_strict, ids_strict = detect_id_in_roi_variants(roi)
            
            if ids_strict is not None:
                for m_corners, m_id in zip(corners_strict, ids_strict):
                    mid_val = int(m_id[0])
                    
                    # Tọa độ trong ROI + Offset ảnh gốc
                    pts_roi = m_corners[0]
                    x, y = float(pts_roi[0][0] + offset_x), float(pts_roi[0][1] + offset_y)
                
                    if mid_val not in final_results:
                        final_results[mid_val] = []
                    
                    
                    should_add = True

                    for old_x, old_y in final_results[mid_val]:
                        dist = ((x - old_x) ** 2 + (y - old_y) ** 2) ** 0.5

                        if dist < DUPLICATE_DISTANCE_THRESHOLD:
                            should_add = False
                            break

                    if should_add:
                        #print(f"Found ID {mid_val} in candidate region with CNN prob {prob:.2f} at position ({x:.2f}, {y:.2f})")
                        final_results[mid_val].append((x, y))

    output_list = []
    for mid in sorted(final_results.keys()):
        for x, y in final_results[mid]:
            output_list.extend([str(mid),f"{x:.3f}",f"{y:.3f}"])
    
    return " ".join(output_list)

if __name__ == "__main__":
    MODEL_PATH = "models/aruco_classifier_model_epoch50.keras"
    TEST_IMAGE = "data/aruco_data/train/000000022087.jpg"
    
    cnn_model = load_cnn_model(MODEL_PATH)
    result = run_aruco_pipeline(TEST_IMAGE, cnn_model)
    print("\n--- FINAL OUTPUT STRING ---")
    print(result if result else "NO MARKERS FOUND")