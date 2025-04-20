import os
import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from rembg import remove
from sklearn.cluster import KMeans
from collections import Counter

def get_dominant_foreground_color(rgba_image: np.ndarray, k=3) -> tuple:
    try:
        if rgba_image.shape[2] != 4:
            return (128, 128, 128)

        alpha = rgba_image[:, :, 3]
        mask = alpha > 0
        rgb_pixels = rgba_image[:, :, :3][mask]

        if len(rgb_pixels) < 10:
            return (128, 128, 128)

        clt = KMeans(n_clusters=k, n_init='auto')
        labels = clt.fit_predict(rgb_pixels)
        counts = Counter(labels)
        center_colors = clt.cluster_centers_
        dominant_color = center_colors[counts.most_common(1)[0][0]]
        return tuple(map(int, dominant_color))
    except Exception:
        return (128, 128, 128)

def clean_edges_with_advanced_mask(img_cv: np.ndarray) -> np.ndarray:
    # Extract alpha channel
    alpha = img_cv[:, :, 3]
    _, fg_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # Step 1: Light region suppression
    gray = cv2.cvtColor(img_cv[:, :, :3], cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center_zone = gray[h//4: 3*h//4, w//4: 3*w//4]
    _, light_center_mask = cv2.threshold(center_zone, 220, 255, cv2.THRESH_BINARY)
    white_patch_mask = np.zeros_like(gray)
    white_patch_mask[h//4: 3*h//4, w//4: 3*w//4] = light_center_mask
    fg_mask_cleaned = cv2.bitwise_and(fg_mask, cv2.bitwise_not(white_patch_mask))

    # Step 2: Morphology + feathering
    kernel = np.ones((3, 3), np.uint8)
    fg_mask_cleaned = cv2.morphologyEx(fg_mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask_cleaned = cv2.dilate(fg_mask_cleaned, kernel, iterations=1)
    fg_mask_cleaned = cv2.GaussianBlur(fg_mask_cleaned, (1, 1), 0)

    # Step 3: Suppress RGB edges near transparent pixels
    soft_mask = fg_mask_cleaned.astype(np.float32) / 255.0
    inverse_mask = 1.0 - soft_mask

    for c in range(3):  # RGB
        img_cv[:, :, c] = img_cv[:, :, c].astype(np.float32) * soft_mask

    img_cv[:, :, :3] = np.clip(img_cv[:, :, :3], 0, 255).astype(np.uint8)
    img_cv[:, :, 3] = fg_mask_cleaned

    # Step 4: Hard clamp edge RGB if alpha is near-zero (final cleanup)
    near_zero = img_cv[:, :, 3] < 15
    img_cv[near_zero, 0:3] = 0  # Set R,G,B to black where alpha ≈ 0

    return img_cv
 
def remove_background_and_generate_versions(
    input_path: str,
    output_no_bg_path: str,
    output_color_bg_path: str,
    resize_width=1024
):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    try:
        image = Image.open(input_path).convert("RGBA")
    except UnidentifiedImageError:
        print(f"❌ Cannot open image file: {input_path}")
        return

    # image.thumbnail((resize_width, resize_width), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    input_data = buffer.getvalue()

    try:
        output_data = remove(input_data, alpha_matting=False)
    except Exception as e:
        print(f"❌ Background removal failed: {str(e)}")
        return

    nparr = np.frombuffer(output_data, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img_cv is None or img_cv.shape[2] < 4:
        print("❌ Failed to decode or missing alpha channel.")
        return

    img_cv = clean_edges_with_advanced_mask(img_cv)

    # Save transparent PNG
    cv2.imwrite(output_no_bg_path, img_cv)
    print(f"✅ Transparent image saved: {output_no_bg_path}")

    # Get dominant color from cleaned foreground
    alpha_channel = img_cv[:, :, 3]
    mask = (alpha_channel > 0).astype(np.uint8) * 255
    dominant_color = get_dominant_foreground_color(img_cv)

    # Replace background
    contrast_color = get_contrast_color(dominant_color)
    bg = np.full_like(img_cv, contrast_color + (255,))
    alpha = mask.astype(float) / 255
    alpha = alpha[:, :, np.newaxis]
    img_colored = (img_cv[:, :, :3] * alpha + bg[:, :, :3] * (1 - alpha)).astype(np.uint8)

    cv2.imwrite(output_color_bg_path, img_colored)
    print(f"✅ Color background image saved: {output_color_bg_path} using {dominant_color}")

def get_contrast_color(rgb: tuple) -> tuple:
    r, g, b = rgb

    # Option 1: Opposite on RGB wheel (simple invert)
    opp_color = (255 - r, 255 - g, 255 - b)

    # Option 2: Check luminance and return white or black for max contrast
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > 186:
        return (0, 0, 0)  # Dark background for light objects
    else:
        return (255, 255, 255)  # Light background for dark objects


def process_image_file_1(input_path: str, output_folder: str) -> dict:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract base name and build output paths
    filename = os.path.basename(input_path)
    base_name, _ = os.path.splitext(filename)
    output_no_bg_path = os.path.join(output_folder, f"{base_name}_nobg.png")
    output_color_bg_path = os.path.join(output_folder, f"{base_name}_colored_bg.jpg")

    # Run background removal
    remove_background_and_generate_versions(
        input_path=input_path,
        output_no_bg_path=output_no_bg_path,
        output_color_bg_path=output_color_bg_path
    )

    # Return useful metadata
    return {
        'original_filename': filename,
        'no_bg_path': output_no_bg_path,
        'color_bg_path': output_color_bg_path
    }

def process_image_file(input_path: str, output_folder: str) -> dict:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(input_path)
    base_name, _ = os.path.splitext(filename)
    output_no_bg_path = os.path.join(output_folder, f"{base_name}_nobg.png")
    output_color_bg_path = os.path.join(output_folder, f"{base_name}_colored_bg.jpg")

    remove_background_and_generate_versions(
        input_path=input_path,
        output_no_bg_path=output_no_bg_path,
        output_color_bg_path=output_color_bg_path
    )

    return {
        'original_filename': filename,
        'no_bg_path': output_no_bg_path,
        'color_bg_path': output_color_bg_path
    }