"""
GreenscreenAnalyzer.py

Dieses Skript bietet eine GUI, mit der Benutzer die Erkennung von Greenscreen-Bereichen in Bildern überprüfen können.
Das erste Bild im Eingabeordner wird als Hintergrund verwendet. Der Greenscreen in allen anderen Bildern wird durch
den Hintergrund ersetzt, und die Konturen werden gezeichnet, um die Erkennung zu visualisieren.

Funktionen:
- create_greenscreen_mask: Erstellt eine Maske für den Greenscreen-Bereich.
- find_greenscreen_contours: Findet die Konturen des Greenscreen-Bereichs.
- replace_greenscreen: Ersetzt den Greenscreen-Bereich durch den Hintergrund.
- analyze_greenscreen: Analysiert ein Bild und ersetzt den Greenscreen.
- select_input_directory: Öffnet einen Dialog zur Auswahl des Eingabeordners.
- select_output_directory: Öffnet einen Dialog zur Auswahl des Ausgabeverzeichnisses.
- process_images: Verarbeitet alle Bilder im Eingabeordner.

Benutzung:
1. Führen Sie das Skript aus.
2. Wählen Sie den Eingabeordner und den Ausgabeverzeichnis.
3. Klicken Sie auf "Process Images", um die Verarbeitung zu starten.
4. Überprüfen Sie die generierten Bilder im Ausgabeverzeichnis, um die Erkennung der Greenscreen-Bereiche zu validieren.
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

def create_greenscreen_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def find_greenscreen_contours(image: np.ndarray) -> list:
    mask = create_greenscreen_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def replace_greenscreen(original_img: np.ndarray, background_img: np.ndarray, contour: np.ndarray, method: str) -> np.ndarray:
    if method == 'convex_hull':
        hull = cv2.convexHull(contour)
        contour_points = hull
    elif method == 'approx':
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        contour_points = approx
    else:
        return original_img

    mask = np.zeros_like(original_img[:, :, 0])
    cv2.drawContours(mask, [contour_points], -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(contour_points)
    resized_background = cv2.resize(background_img, (w, h), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.bitwise_not(mask[y:y+h, x:x+w])

    for c in range(0, 3):
        original_img[y:y+h, x:x+w, c] = (
            resized_background[:, :, c] * (mask[y:y+h, x:x+w] / 255.0) +
            original_img[y:y+h, x:x+w, c] * (mask_inv / 255.0)
        )

    return original_img

def analyze_greenscreen(image_path: str, background_img: np.ndarray, output_dir: str, min_area: int = 10000):
    if not os.path.exists(image_path) or not os.access(image_path, os.R_OK):
        print(f"Image path does not exist or is not readable: {image_path}")
        return
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    contours = find_greenscreen_contours(image)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(image, [hull], -1, (0, 0, 255), 2)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(image, [approx], -1, (0, 255, 255), 2)
            
            result_convex_hull = replace_greenscreen(image.copy(), background_img, contour, method='convex_hull')
            result_convex_hull_path = os.path.join(output_dir, f'result_convex_hull_{os.path.basename(image_path)}')
            cv2.imwrite(result_convex_hull_path, result_convex_hull)
            print(f"Result image with convex hull saved to {result_convex_hull_path}")
            
            result_approx = replace_greenscreen(image.copy(), background_img, contour, method='approx')
            result_approx_path = os.path.join(output_dir, f'result_approx_{os.path.basename(image_path)}')
            cv2.imwrite(result_approx_path, result_approx)
            print(f"Result image with approximate polygon saved to {result_approx_path}")
    
    result_path = os.path.join(output_dir, 'result_' + os.path.basename(image_path))
    cv2.imwrite(result_path, image)
    print(f"Result image saved to {result_path}")

def select_input_directory():
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    input_dir_var.set(input_dir)

def select_output_directory():
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    output_dir_var.set(output_dir)

def process_images():
    input_dir = input_dir_var.get()
    output_dir = output_dir_var.get()

    if not input_dir or not output_dir:
        messagebox.showerror("Error", "Please select all the required directories.")
        return
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < 2:
        messagebox.showerror("Error", "The input directory must contain at least two images.")
        return

    background_path = os.path.join(input_dir, image_files[0])
    background_img = cv2.imread(background_path)

    if background_img is None:
        messagebox.showerror("Error", f"Failed to load background image: {background_path}")
        return

    image_paths = [os.path.join(input_dir, f) for f in image_files[1:]]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_path in image_paths:
        analyze_greenscreen(image_path, background_img, output_dir)

    messagebox.showinfo("Success", "Processing completed.")

# GUI setup
root = tk.Tk()
root.title("Greenscreen Analyzer")

input_dir_var = tk.StringVar()
output_dir_var = tk.StringVar()

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

tk.Label(frame, text="Input Directory:").grid(row=0, column=0, sticky="e")
tk.Entry(frame, textvariable=input_dir_var, width=50).grid(row=0, column=1)
tk.Button(frame, text="Browse", command=select_input_directory).grid(row=0, column=2)

tk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky="e")
tk.Entry(frame, textvariable=output_dir_var, width=50).grid(row=1, column=1)
tk.Button(frame, text="Browse", command=select_output_directory).grid(row=1, column=2)

tk.Button(frame, text="Process Images", command=process_images).grid(row=2, columnspan=3, pady=10)

root.mainloop()
