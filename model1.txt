import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color, measure
import csv
import os

# --- Create output directory ---
output_dir = "output2"
os.makedirs(output_dir, exist_ok=True)

# --- Read the image in color (24-bit: 3 channels, 8 bits each) ---
img = cv2.imread("image/grain2.jpg", cv2.IMREAD_COLOR)   # (h, w, 3), dtype=uint8

# Pixel-to-micrometer conversion (example: 1 px = 12 µm)
pixels_to_um = 12

# --- Convert to grayscale for thresholding ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Otsu's thresholding ---
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Morphological cleanup ---
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

# --- Create boolean mask (True = grain) ---
mask = dilated == 255

# --- Label connected regions (grains) ---
s = [[1,1,1],[1,1,1],[1,1,1]]
labeled_mask, num_labels = ndimage.label(mask, structure=s)

# --- Colored visualization of grains ---
img2 = color.label2rgb(labeled_mask, bg_label=0)

# --- Measure properties of each grain ---
clusters = measure.regionprops(labeled_mask, gray)

# --- Grain size classification function (Wentworth scale in µm) ---
def classify_grain(size_um):
    if size_um < 62.5:
        return "Silt or smaller"
    elif size_um < 125:
        return "Very Fine Sand"
    elif size_um < 250:
        return "Fine Sand"
    elif size_um < 500:
        return "Medium Sand"
    elif size_um < 1000:
        return "Coarse Sand"
    else:
        return "Very Coarse Sand"

# --- Open CSV file to save results ---
csv_path = os.path.join(output_dir, 'grain_measurements.csv')
with open(csv_path, 'w', newline='') as output_file:
    writer = csv.writer(output_file)

    # Write header
    writer.writerow([
        'Label', 'Area (µm^2)', 'EquivalentDiameter (µm)',
        'MajorAxisLength (µm)', 'MinorAxisLength (µm)',
        'Perimeter (µm)', 'Orientation (deg)',
        'AspectRatio', 'Circularity', 'Solidity',
        'Grain Class'
    ])

    grain_sizes = []  # for histogram

    # Loop through grains
    for cluster_props in clusters:
        label = cluster_props.label

        # Extract raw measurements
        area = cluster_props.area * (pixels_to_um ** 2)  # µm²
        eq_diam = cluster_props.equivalent_diameter * pixels_to_um
        major = cluster_props.major_axis_length * pixels_to_um
        minor = cluster_props.minor_axis_length * pixels_to_um
        perimeter = cluster_props.perimeter * pixels_to_um
        orientation = cluster_props.orientation * 57.2958  # rad → deg
        convex_area = cluster_props.convex_area * (pixels_to_um ** 2)

        # Derived features
        aspect_ratio = major / minor if minor > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        solidity = area / convex_area if convex_area > 0 else 0

        # Grain classification
        grain_class = classify_grain(eq_diam)

        # Save to CSV
        writer.writerow([
            label, area, eq_diam, major, minor,
            perimeter, orientation, aspect_ratio,
            circularity, solidity, grain_class
        ])

        # Collect sizes for histogram
        grain_sizes.append(eq_diam)

# --- Show and save visualization ---
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Labeled Grains")
plt.imshow(img2)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grain_labels.png"), dpi=300)
plt.show()

# --- Plot grain size distribution ---
plt.figure(figsize=(6, 4))
plt.hist(grain_sizes, bins=15, color='steelblue', edgecolor='black')
plt.title("Grain Size Distribution")
plt.xlabel("Equivalent Diameter (µm)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "grain_size_distribution.png"), dpi=300)
plt.show()

print(f"✅ Results saved in folder: {output_dir}")
