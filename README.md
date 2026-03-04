# Measuring Object Size Using Python & OpenCV

By **Bakhtiyar Saeed**  
Subject: Digital Image Processing (DIP) — BS IT 8th Semester

---

## What it does

Loads an image, detects all objects using contour detection, and measures:
- Area (in pixels or real-world cm² if you provide a reference)
- Width and Height of each object's bounding box

Outputs a 4-panel image showing each step: Original → Grayscale → Threshold → Measured.

---

## Install

```bash
pip install -r requirements.txt
```

---

## How to Run

### Run on a demo image (no image needed)
```bash
python measure_objects.py --demo
```

### Run on your own image
```bash
python measure_objects.py --image your_image.jpg
```

### Run with real-world measurements
Include a reference object in your image (like a coin or ruler), then tell the script its real width in cm:
```bash
python measure_objects.py --image your_image.jpg --ref-width 2.4
```
If a 2.4cm coin appears in the image, the script uses it to calculate the scale factor and converts all measurements to cm².

### Save output to a custom filename
```bash
python measure_objects.py --image your_image.jpg --output result.jpg
```

---

## How it works

1. Load image with `cv2.imread()`
2. Convert to grayscale with `cv2.cvtColor()`
3. Apply Otsu thresholding to separate objects from background
4. Find contours with `cv2.findContours()`
5. For each contour: calculate area with `cv2.contourArea()`, get bounding box with `cv2.boundingRect()`
6. If a reference object width is provided, compute scale factor = real size / pixel size
7. Draw results and save

---

## Scale Factor Formula

```
scale_factor = real_world_width_cm / pixel_width
real_area_cm² = pixel_area × scale_factor²
```
