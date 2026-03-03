"""
Measuring Object Size Using Python & OpenCV
By: Usman Noor & Bakhtiyar Saeed
Subject: Digital Image Processing (DIP) — BS IT 8th Semester

HOW TO USE:
    python measure_objects.py --image your_image.jpg
    python measure_objects.py --image your_image.jpg --ref-width 2.4   (real-world cm of reference object)
    python measure_objects.py --demo                                     (run on a generated test image)
"""

import cv2
import numpy as np
import argparse
import os


# ─────────────────────────────────────────────
# 1. GENERATE A DEMO IMAGE (so you can test without your own image)
# ─────────────────────────────────────────────

def create_demo_image(path="demo_image.jpg"):
    """Creates a test image with some shapes on a white background."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Draw a few shapes to act as "objects"
    cv2.rectangle(img, (50, 50), (200, 180), (30, 30, 200), -1)       # Blue rectangle
    cv2.circle(img, (380, 120), 80, (30, 180, 30), -1)                 # Green circle
    cv2.ellipse(img, (620, 120), (100, 60), 0, 0, 360, (180, 30, 30), -1)  # Red ellipse
    cv2.rectangle(img, (50, 300), (350, 500), (200, 100, 30), -1)      # Orange rectangle
    cv2.circle(img, (550, 400), 120, (100, 30, 200), -1)               # Purple circle

    cv2.imwrite(path, img)
    print(f"✅ Demo image created: {path}")
    return path


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(img):
    """Convert to grayscale and apply Otsu thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)

    # Otsu threshold — automatically finds the best threshold value
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return gray, thresh


# ─────────────────────────────────────────────
# 3. DETECT CONTOURS & MEASURE
# ─────────────────────────────────────────────

def detect_and_measure(img, thresh, scale_factor=1.0, unit="px²"):
    """
    Finds all object contours, calculates area & dimensions,
    and draws results on the image.

    scale_factor: multiply pixel area by this to get real-world area
    unit: display unit label (e.g. 'cm²' or 'px²')
    """
    result = img.copy()

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    measurements = []

    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)

        # Skip tiny noise contours
        if area_px < 500:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Real-world conversion
        real_area = area_px * (scale_factor ** 2)
        real_w = w * scale_factor
        real_h = h * scale_factor

        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw contour
        cv2.drawContours(result, [cnt], -1, (255, 100, 0), 1)

        # Label
        label1 = f"#{i+1} Area: {real_area:.1f} {unit}"
        label2 = f"W:{real_w:.1f} H:{real_h:.1f}"
        cv2.putText(result, label1, (x, y - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(result, label2, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

        measurements.append({
            "object": i + 1,
            "area_px": area_px,
            "real_area": real_area,
            "width_px": w,
            "height_px": h,
            "real_width": real_w,
            "real_height": real_h,
            "unit": unit
        })

    return result, measurements


# ─────────────────────────────────────────────
# 4. COMPUTE SCALE FACTOR FROM REFERENCE OBJECT
# ─────────────────────────────────────────────

def compute_scale(thresh, ref_real_width_cm):
    """
    Finds the largest contour (assumed to be the reference object),
    measures its pixel width, and returns the scale factor.

    scale_factor = real_width_cm / pixel_width
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 1.0

    # Use the largest contour as the reference
    largest = max(contours, key=cv2.contourArea)
    _, _, w, _ = cv2.boundingRect(largest)

    if w == 0:
        return 1.0

    scale = ref_real_width_cm / w
    print(f"📏 Reference object width: {w}px → scale factor: {scale:.4f} cm/px")
    return scale


# ─────────────────────────────────────────────
# 5. SAVE & DISPLAY RESULTS
# ─────────────────────────────────────────────

def save_and_show(original, gray, thresh, result, output_path="output.jpg"):
    """Saves a 4-panel result image showing each processing step."""
    # Resize all to same height
    h, w = original.shape[:2]

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Add step labels
    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        return out

    panel = np.hstack([
        label(original, "1. Original"),
        label(gray_bgr, "2. Grayscale"),
        label(thresh_bgr, "3. Threshold"),
        label(result, "4. Measured"),
    ])

    # Scale down if too wide
    max_width = 1600
    if panel.shape[1] > max_width:
        scale = max_width / panel.shape[1]
        panel = cv2.resize(panel, (max_width, int(panel.shape[0] * scale)))

    cv2.imwrite(output_path, panel)
    print(f"✅ Result saved: {output_path}")
    return panel


def print_results(measurements):
    """Print a clean table of measurements to the terminal."""
    if not measurements:
        print("⚠️  No objects detected.")
        return

    print("\n" + "="*60)
    print(f"{'Object':<8} {'Area':>12} {'Width':>10} {'Height':>10}")
    print("="*60)
    for m in measurements:
        unit_label = m['unit'].replace('²','²')
        print(f"  #{m['object']:<5} {m['real_area']:>10.1f} {unit_label}   "
              f"{m['real_width']:>8.1f}   {m['real_height']:>8.1f}")
    print("="*60)
    print(f"Total objects detected: {len(measurements)}\n")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Measure object sizes using OpenCV")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--ref-width", type=float, default=None,
                        help="Real-world width of reference object in cm (optional)")
    parser.add_argument("--demo", action="store_true",
                        help="Run on a generated demo image")
    parser.add_argument("--output", type=str, default="output.jpg",
                        help="Output image path (default: output.jpg)")
    args = parser.parse_args()

    # Determine image path
    if args.demo:
        image_path = create_demo_image("demo_image.jpg")
    elif args.image:
        image_path = args.image
    else:
        print("No image provided. Running in demo mode...")
        image_path = create_demo_image("demo_image.jpg")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"\n📷 Image loaded: {image_path} ({img.shape[1]}x{img.shape[0]}px)")

    # Preprocess
    gray, thresh = preprocess(img)

    # Scale factor
    if args.ref_width:
        scale_factor = compute_scale(thresh, args.ref_width)
        unit = "cm²"
    else:
        scale_factor = 1.0
        unit = "px²"
        print("ℹ️  No reference width given — showing pixel measurements.")
        print("   Use --ref-width 2.4 to convert to real-world units.\n")

    # Measure
    result, measurements = detect_and_measure(img, thresh, scale_factor, unit)

    # Print results
    print_results(measurements)

    # Save output
    save_and_show(img, gray, thresh, result, args.output)
    print(f"✅ Done! Open '{args.output}' to see the result.\n")


if __name__ == "__main__":
    main()
