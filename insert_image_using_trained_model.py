import os
import numpy as np
import cv2
from typing import Union

def create_greenscreen_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Rauschunterdrückung anwenden
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Kleine weiße Flecken entfernen
    num_labels, labels_im = cv2.connectedComponents(mask)
    
    # Komponenten analysieren und die größte Komponente behalten
    max_label = 1
    max_size = 0
    for label in range(1, num_labels):
        size = np.sum(labels_im == label)
        if size > max_size:
            max_size = size
            max_label = label

    mask = np.uint8(labels_im == max_label) * 255
    
    return mask

def replace_greenscreen(original_img: np.ndarray, background_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(mask)
    print(f"Greenscreen area - Width: {w} px, Height: {h} px")
    
    # Hintergrundbild auf die Größe des Greenscreen-Bereichs strecken
    background_img_resized = cv2.resize(background_img, (w, h), interpolation=cv2.INTER_AREA)

    # Erstellung des Ergebnisbildes
    result = original_img.copy()
    mask_cropped = mask[y:y+h, x:x+w]
    mask_inv = cv2.bitwise_not(mask_cropped)

    for c in range(0, 3):
        result[y:y+h, x:x+w, c] = (
            background_img_resized[:, :, c] * (mask_cropped / 255.0) + 
            result[y:y+h, x:x+w, c] * (mask_inv / 255.0)
        )

    return result

def main():
    original_image_path: Union[str, None] = '128.jpg'  # Updated to the correct file path
    background_image_path: Union[str, None] = '81.jpg'  # Make sure this path is correct for your setup
    output_image_path: Union[str, None] = '128g.jpg'

    try:
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Original image file '{original_image_path}' not found.")
        if not os.path.exists(background_image_path):
            raise FileNotFoundError(f"Background image file '{background_image_path}' not found.")
        
        original_img = cv2.imread(original_image_path)
        background_img = cv2.imread(background_image_path)

        if original_img is None:
            raise ValueError(f"Failed to load original image from '{original_image_path}'.")
        if background_img is None:
            raise ValueError(f"Failed to load background image from '{background_image_path}'.")

        mask = create_greenscreen_mask(original_img)
        
        result = replace_greenscreen(original_img, background_img, mask)

        cv2.imwrite(output_image_path, result)
        print(f'Result saved to {output_image_path}')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
