import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocess import generate_augmented_images
from src.detector import extract_candidates, filter_and_cluster_candidates, is_valid_geometry_loose

TRAIN_IMAGE_DIR = "data/aruco_data/train"
GT_CSV_PATH = "data/train.csv"
SAVE_REAL_DIR = "data/train/real"
SAVE_FAKE_DIR = "data/train/fake"

PATCH_SIZE = 128
DISTANCE_THRESHOLD = 2

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_gt_string(prediction_string):
    """Parse chuỗi format: 'id x y id x y' thành list các dict"""
    if pd.isna(prediction_string) or prediction_string == "":
        return []
    tokens = prediction_string.strip().split()
    results = []
    for i in range(0, len(tokens), 3):
        results.append({
            "id": int(tokens[i]),
            "top_left": (float(tokens[i+1]), float(tokens[i+2]))
        })
    return results

def warp_marker_patch(image, corners, size=64):
    """Warp vùng candidate về ảnh vuông size x size"""
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)
    src = np.array(corners, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (size, size))

def is_valuable_fake(corners, img, threshold=175):
    h_img, w_img = img.shape[:2]
    pts = np.array(corners, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    
    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(w_img, x + w), min(h_img, y + h)
    
    if x_end <= x_start or y_end <= y_start: return False
    
    patch = img[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score > threshold

def contains_any_gt_top_left(corners, gt_list):
    pts = np.array(corners, dtype=np.int32)
    
    for gt in gt_list:
        x, y = gt["top_left"]
        if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
            return True
        
    return False

def main():
    ensure_dir(SAVE_REAL_DIR)
    ensure_dir(SAVE_FAKE_DIR)
    
    df = pd.read_csv(GT_CSV_PATH, dtype={"image_id": str})
    real_count = 0
    fake_count = 0
    total_missed = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Dataset"):
        image_id = str(row["image_id"])
        gt_list = parse_gt_string(row["prediction_string"])
        count_r = 0
        count_f = 0
        
        img_path = os.path.join(TRAIN_IMAGE_DIR, f"{image_id}.jpg")
        image = cv2.imread(img_path)
        if image is None: continue
        
        aug_dict = generate_augmented_images(image)
        all_raw_candidates = extract_candidates(image, aug_dict)
        refined_candidates = filter_and_cluster_candidates(all_raw_candidates)
        
        gt_matched = [False] * len(gt_list)
        
        for cand in refined_candidates:
            source_name = cand["source"]
            target_data = aug_dict[source_name]
            target_img = target_data["image"]
            t_info = target_data.get("transform_info")
            
            cand_corners = cand["corners"]
            top_left = cand_corners[0]
            
            is_real = False
            for i, gt in enumerate(gt_list):
                gt_pos = np.array(gt["top_left"])
                
                for corner_pt in cand_corners:
                    dist = np.linalg.norm(corner_pt - gt_pos)
                    if dist < DISTANCE_THRESHOLD:
                        is_real = True
                        gt_matched[i] = True
                        break
                        
                if is_real: break
            
            
            if t_info is not None:
                crop_corners = cand["raw_corners"]
            else:
                crop_corners = cand["corners"]
            patch = warp_marker_patch(target_img, crop_corners, PATCH_SIZE)

            if is_real:
                save_name = f"real_{image_id}_{real_count}.png"
                print(source_name, top_left)
                cv2.imwrite(os.path.join(SAVE_REAL_DIR, save_name), patch)
                real_count += 1
                count_r += 1
            else:
                if is_valid_geometry_loose(crop_corners, 300) and is_valuable_fake(crop_corners, target_img) and cand["votes"] > 5 and not contains_any_gt_top_left(crop_corners, gt_list):
                    save_name = f"fake_{image_id}_{fake_count}.png"
                    cv2.imwrite(os.path.join(SAVE_FAKE_DIR, save_name), patch)
                    fake_count += 1
                    count_f += 1
                
        for i, matched in enumerate(gt_matched):
            if not matched:
                total_missed += 1
                missed_id = gt_list[i]["id"]
                missed_pos = gt_list[i]["top_left"]
                # In ra cảnh báo lỗi
                tqdm.write(f"[MISSED] Image {image_id}: Marker ID {missed_id} at {missed_pos} was NOT detected!")
        tqdm.write(f"DONE Image {image_id}: {count_r} image on {len(gt_matched)} real marker, {count_f} fake marker")

    print(f"\nHoàn tất!")
    print(f"Real patches: {real_count}")
    print(f"Fake patches: {fake_count}")
    if total_missed > 0:
        print(f"TỔNG CỘNG BỊ BỎ SÓT: {total_missed} markers")
    else:
        print(f"Tuyệt vời! Không bỏ sót bất kỳ marker nào từ Ground Truth.")

if __name__ == "__main__":
    main()