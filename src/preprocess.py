import cv2
import numpy as np
import random

# --- CÁC HÀM BỔ TRỢ (Helper Functions) ---

def aug_clahe_gamma_sharp(img, gamma=3.2, clahe_clip=4.5, clahe_grid=8, sharp=1.7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    gamma_corr = np.power(gray_f, 1.0 / gamma)
    gray_gamma = np.uint8(np.clip(gamma_corr * 255, 0, 255))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    gray_clahe = clahe.apply(gray_gamma)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)
    sharp_img = cv2.addWeighted(gray_clahe, 1 + sharp, blur, -sharp, 0)
    return cv2.cvtColor(sharp_img, cv2.COLOR_GRAY2BGR)

def aug_motion_blur(img, ksize=11):
    kernel = np.zeros((ksize, ksize))
    if random.random() > 0.5:
        kernel[int((ksize-1)/2), :] = np.ones(ksize)
    else:
        kernel[:, int((ksize-1)/2)] = np.ones(ksize)
    kernel /= ksize
    return cv2.filter2D(img, -1, kernel)

def aug_gaussian_noise(img):
    row, col, ch = img.shape
    sigma = random.uniform(10, 30)
    gauss = np.random.normal(0, sigma, (row, col, ch))
    noisy = np.clip(img.astype(np.float32) + gauss, 0, 255)
    return noisy.astype(np.uint8)

def aug_perspective_tilt(img):
    h, w = img.shape[:2]
    src = np.float32([[0,0], [w,0], [0,h], [w,h]])
    
    offset_x = random.uniform(0.05, 0.15) * w
    offset_y = random.uniform(0.05, 0.15) * h
    
    dst = np.float32([
        [random.uniform(0, offset_x), random.uniform(0, offset_y)],
        [w - random.uniform(0, offset_x), random.uniform(0, offset_y)],
        [random.uniform(0, offset_x), h - random.uniform(0, offset_y)],
        [w - random.uniform(0, offset_x), h - random.uniform(0, offset_y)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    tilted = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))
    
    M_inv = np.linalg.inv(M)
    transform_info = {
        "type": "matrix",
        "M_inv": M_inv
    }
    return tilted, transform_info

def aug_dilation(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def aug_erosion(img):
    kernel = np.ones((2,2), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def aug_zoom_in(img, scale_range=(1.2, 1.8)):
    h, w = img.shape[:2]
    # Random tỷ lệ zoom
    scale = random.uniform(scale_range[0], scale_range[1])
    
    new_w = int(w / scale)
    new_h = int(h / scale)
    
    # Random vị trí crop
    x_offset = random.randint(0, w - new_w)
    y_offset = random.randint(0, h - new_h)
    
    cropped = img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    transform_info = {
        "type": "crop_resize",
        "x_offset": x_offset,
        "y_offset": y_offset,
        "scale_x": w / new_w,
        "scale_y": h / new_h
    }
    return zoomed, transform_info

def aug_scale_down(img, scale_range=(0.5, 0.9)):
    h, w = img.shape[:2]
    scale = random.uniform(scale_range[0], scale_range[1])
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize nhỏ lại để mất chi tiết (pixelation)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    transform_info = {
        "type": "crop_resize",
        "x_offset": 0,
        "y_offset": 0,
        "scale_x": new_w / w,
        "scale_y": new_h / h
    }
    return scaled, transform_info

def aug_occlusion(img, occ_ratio_range=(0.05, 0.2)):
    """Tạo các hình khối đè lên ảnh để giả lập vật cản"""
    h, w = img.shape[:2]
    res = img.copy()
    # Tạo từ 1 đến 3 vùng che khuất
    for _ in range(random.randint(1, 3)):
        occ_w = int(w * random.uniform(*occ_ratio_range))
        occ_h = int(h * random.uniform(*occ_ratio_range))
        x = random.randint(0, w - occ_w)
        y = random.randint(0, h - occ_h)
        # Màu ngẫu nhiên (đen, trắng hoặc xám)
        color = random.choice([(0,0,0), (255,255,255), (128,128,128)])
        cv2.rectangle(res, (x, y), (x + occ_w, y + occ_h), color, -1)
    return res

def aug_background_clutter(img):
    """Gây nhiễu nền bằng cách vẽ các đường line hoặc vòng tròn ngẫu nhiên"""
    h, w = img.shape[:2]
    res = img.copy()
    for _ in range(10):
        pt1 = (random.randint(0, w), random.randint(0, h))
        pt2 = (random.randint(0, w), random.randint(0, h))
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.line(res, pt1, pt2, color, random.randint(1, 3))
    return res

# --- DICTIONARY CHÍNH ---

AUGMENTATIONS = {
    "original": lambda img: img.copy(),
    "median_blur": lambda img: cv2.medianBlur(img, 5),
    "clahe_sharp": lambda img: aug_clahe_gamma_sharp(img),
    "high_contrast": lambda img: cv2.convertScaleAbs(img, alpha=1.8, beta=-40),
    "low_contrast": lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=50),
    "bright_over": lambda img: cv2.convertScaleAbs(img, alpha=4.2, beta=175),
    "dark_under": lambda img: cv2.convertScaleAbs(img, alpha=0.2, beta=-100),
    "noise_gauss": lambda img: aug_gaussian_noise(img),
    "dilate": lambda img: aug_dilation(img),
    "erode": lambda img: aug_erosion(img),
    "invert": lambda img: cv2.bitwise_not(img),
    "heavy_blur": lambda img: cv2.GaussianBlur(img, (9, 9), 0),
    "motion_blur": lambda img: aug_motion_blur(img, ksize=11),
    "scale_down": lambda img: aug_scale_down(img, scale_range=(0.3, 0.8)),
    "perspective": lambda img: aug_perspective_tilt(img),
    "zoom_in": lambda img: aug_zoom_in(img),
    "background_clutter": lambda img: aug_background_clutter(img),
    "occlusion": lambda img: aug_occlusion(img, occ_ratio_range=(0.05, 0.15)),
}

def generate_augmented_images(image_bgr):
    """Trả về một dictionary chứa cấu trúc chuẩn: {"image": img, "transform_info": info}"""
    aug_images = {}
    for name, func in AUGMENTATIONS.items():
        result = func(image_bgr.copy())
        
        if isinstance(result, tuple):
            img, t_info = result
        else:
            img = result
            t_info = None # Các phép biến đổi màu sắc/mờ không làm thay đổi tọa độ
            
        aug_images[name] = {
            "image": img,
            "transform_info": t_info
        }
    return aug_images