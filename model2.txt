import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color, measure, morphology, exposure
import csv
import os

# --- Output directory ---
output_dir = "output_refined"
os.makedirs(output_dir, exist_ok=True)

# --- Load image ---
img = cv2.imread("image/grain1.jpg", cv2.IMREAD_COLOR)  # Update path if needed
pixels_to_um = 12   # Example scale: 1 pixel = 12 µm

# --- Preprocessing ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply CLAHE for local contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray_eq, (5,5), 0)

# Combine Otsu + Adaptive thresholding
_, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 2)
thresh = cv2.bitwise_or(otsu, adaptive)

# Morphological opening + closing
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Remove very small objects (noise)
mask = closing > 0
mask = morphology.remove_small_objects(mask, min_size=50)  # px² threshold
mask = mask.astype(np.uint8) * 255

# --- Distance transform for watershed markers ---
dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
dist = cv2.GaussianBlur(dist, (3,3), 0)  # smooth distance map

# Normalize distance for h-maxima
dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

# Threshold for sure foreground
_, sure_fg = cv2.threshold(dist_norm, 0.3, 1.0, cv2.THRESH_BINARY)
sure_fg = morphology.binary_opening(sure_fg, morphology.disk(2))
sure_fg = np.uint8(sure_fg) * 255

# Sure background
sure_bg = cv2.dilate(mask, kernel, iterations=3)

# Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# --- Connected components for watershed markers ---
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# --- Apply watershed ---
markers = cv2.watershed(img, markers)

# --- Create colored label map ---
labeled_mask = np.where(markers > 1, markers, 0)
colored = color.label2rgb(labeled_mask, bg_label=0, bg_color=(0,0,0))
colored = cv2.cvtColor((colored * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Draw thin contours (1px border)
contours, _ = cv2.findContours((markers == -1).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(colored, contours, -1, (255, 0, 0), 1)

# --- Measure properties ---
clusters = measure.regionprops(labeled_mask, gray)

# Grain classification (Wentworth scale, µm)
def classify_grain(d):
    if d < 62.5: return "Silt/Clay"
    elif d < 125: return "Very Fine Sand"
    elif d < 250: return "Fine Sand"
    elif d < 500: return "Medium Sand"
    elif d < 1000: return "Coarse Sand"
    elif d < 2000: return "Very Coarse Sand"
    else: return "Granule"

csv_path = os.path.join(output_dir, "grain_measurements.csv")
grain_sizes = []

with open(csv_path, 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow([
        "Label", "Area (µm^2)", "EquivalentDiameter (µm)",
        "MajorAxisLength (µm)", "MinorAxisLength (µm)",
        "Perimeter (µm)", "Orientation (deg)",
        "AspectRatio", "Circularity", "Solidity", "Class"
    ])

    for cluster_props in clusters:
        label = cluster_props.label
        area = cluster_props.area * (pixels_to_um ** 2)
        eq_diam = cluster_props.equivalent_diameter * pixels_to_um
        major = cluster_props.major_axis_length * pixels_to_um
        minor = cluster_props.minor_axis_length * pixels_to_um
        perimeter = cluster_props.perimeter * pixels_to_um
        orientation = cluster_props.orientation * 57.2958
        convex_area = cluster_props.convex_area * (pixels_to_um ** 2)

        aspect_ratio = major / minor if minor > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        solidity = area / convex_area if convex_area > 0 else 0

        # Filter false grains (remove too small/too elongated shapes)
        if area < 20 or circularity < 0.2 or solidity < 0.5:
            continue

        category = classify_grain(eq_diam)

        writer.writerow([
            label, area, eq_diam, major, minor, perimeter,
            orientation, aspect_ratio, circularity, solidity, category
        ])

        grain_sizes.append(eq_diam)

# --- Save visualizations ---
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Refined Grain Segmentation")
plt.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grain_labels.png"), dpi=300)
plt.show()

# --- Histogram ---
plt.figure(figsize=(6,4))
plt.hist(grain_sizes, bins=15, color='steelblue', edgecolor='black')
plt.title("Grain Size Distribution")
plt.xlabel("Equivalent Diameter (µm)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "grain_size_distribution.png"), dpi=300)
plt.show()

print(f"✅ Results saved in folder: {output_dir}")
