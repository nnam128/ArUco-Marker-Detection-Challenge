import cv2

def load_image(image_path):
    """
    Load image from path.
    Returns:
        image (BGR)
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    return image


def resize_if_needed(image, max_size=1280):
    """
    Optional speed optimization.
    Resize large images while preserving aspect ratio.
    """
    h, w = image.shape[:2]
    
    longest = max(h, w)
    
    if longest <= max_size:
        return image
    
    scale = max_size / longest
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )
    
    return resized