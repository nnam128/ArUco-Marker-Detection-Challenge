import cv2
import numpy as np

def build_loose_candidate_proposer():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36H12)
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.08
    params.minCornerDistanceRate = 0.01
    params.minDistanceToBorder = 1
    params.minOtsuStdDev = 0
    params.errorCorrectionRate = 0.5
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 5
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)

def map_corners_to_original(corners, transform_info):
    if transform_info is None or "type" not in transform_info:
        return corners

    aug_type = transform_info["type"]

    if aug_type == "matrix":
        M_inv = transform_info["M_inv"]
        # Đảm bảo array contiguous và là kiểu float32 để tránh lỗi ngầm của OpenCV
        corners_reshaped = np.ascontiguousarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        corners_mapped = cv2.perspectiveTransform(corners_reshaped, M_inv)
        return corners_mapped.reshape(4, 2)

    elif aug_type == "crop_resize":
        x_offset = transform_info["x_offset"]
        y_offset = transform_info["y_offset"]
        scale_x = transform_info["scale_x"]
        scale_y = transform_info["scale_y"]
        
        corners_mapped = np.copy(corners).astype(np.float32)
        corners_mapped[:, 0] /= scale_x
        corners_mapped[:, 1] /= scale_y
        corners_mapped[:, 0] += x_offset
        corners_mapped[:, 1] += y_offset
        return corners_mapped

    return corners

def extract_candidates(image_bgr, aug_images_dict):
    detector = build_loose_candidate_proposer()
    all_candidates = []

    for aug_name, aug_data in aug_images_dict.items():
        aug_img = aug_data["image"]
        transform_info = aug_data.get("transform_info", None)
        
        corners, ids, rejected = detector.detectMarkers(aug_img)
        '''
        corners = [
            array([[
                [x1, y1],
                [x2, y2],
                [x3, y3],
                [x4, y4]
            ]], dtype=float32),
        '''
        
        def process_found(raw_corners, c_type):
            for c in raw_corners:
                curr_corners = c[0].copy()
                
                raw_c=curr_corners
                curr_corners = map_corners_to_original(curr_corners, transform_info)
                
                all_candidates.append({
                    "corners": curr_corners,
                    "corners_raw": raw_c,
                    "source": aug_name,
                    "type": c_type
                })
                
        if corners is not None: process_found(corners, "detected")
        if rejected is not None: process_found(rejected, "rejected")
        
    return all_candidates

def is_valid_geometry_loose(corners, min_area=50):
    pts = np.float32(corners)
    
    area = cv2.contourArea(pts)
    if area < min_area: 
        return False
        
    if not cv2.isContourConvex(pts):
        return False
        
    return True

def filter_and_cluster_candidates(all_candidates, dist_thresh=2, min_votes=1):
    if not all_candidates: return []
    clusters = []
    
    for cand in all_candidates:
        corners = cand["corners"]
        
        if not is_valid_geometry_loose(corners):
            continue
            
        top_left = corners[0]
        matched = False
        
        for cluster in clusters:
            dist = np.linalg.norm(top_left - cluster["top_left"])
            if dist < dist_thresh:
                cluster["count"] += 1
                
                
                if (cand.get("type") == "detected" and cluster["type"] != "detected"): #or (cand.get("type") == cluster["type"] and cand_area > current_best_area):
                    cluster["source"] = cand.get("source")
                    cluster["type"] = cand.get("type")
                    cluster["best_corners"] = corners
                    cluster["raw_corners"] = cand.get("corners_raw")
                matched = True
                break
        
        if not matched:
            clusters.append({
                "top_left": top_left,
                "count": 1,
                "best_corners": corners,
                "raw_corners": cand.get("corners_raw"),
                "source": cand.get("source"),
                "type": cand.get("type", "rejected")
            })
            
    clusters.sort(key=lambda x: (x["type"] == "detected"), reverse=True)
    final_candidates = []
    
    def is_inside(corners_a, corners_b):
        """
        Kiểm tra candidate A có nằm trong candidate B hay không
        bằng bounding box đơn giản.
        """
        ax_min, ay_min = corners_a.min(axis=0)
        ax_max, ay_max = corners_a.max(axis=0)

        bx_min, by_min = corners_b.min(axis=0)
        bx_max, by_max = corners_b.max(axis=0)

        return (
            ax_min >= bx_min and
            ay_min >= by_min and
            ax_max <= bx_max and
            ay_max <= by_max
        )
    for cluster in clusters:
        if cluster["count"] < min_votes:
            continue
        
        current_corners = cluster["best_corners"]
        should_skip = False
        
        for existed in final_candidates:
            existed_corners = existed["corners"]
            
            if is_inside(current_corners, existed_corners):
                should_skip = True
                break
            
        if not should_skip:
            final_candidates.append({
                "corners": current_corners,
                "raw_corners": cluster["raw_corners"],
                "votes": cluster["count"],
                "type": cluster["type"],
                "source": cluster["source"]
            })
            
    return final_candidates