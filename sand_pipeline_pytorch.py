import torch
import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, color
import matplotlib.pyplot as plt
import csv
import os
from train_unet_pytorch import UNet  # import model class
from torchvision import transforms
from PIL import Image

# --- Config ---
pixels_to_um = 12.0
output_dir = "output_cnn"
os.makedirs(output_dir, exist_ok=True)

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_sand.pth", map_location=device))
model.eval()

# --- Preprocess image ---
img_path = "image/grain1.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
h, w = img.shape[:2]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
x = transform(pil_img).unsqueeze(0).to(device)

# --- Predict mask ---
with torch.no_grad():
    pred = model(x)[0][0].cpu().numpy()

mask = (pred > 0.5).astype("uint8")
mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# --- Label grains ---
s = [[1,1,1],[1,1,1],[1,1,1]]
labeled_mask, num_labels = ndimage.label(mask_resized, structure=s)
img2 = color.label2rgb(labeled_mask, bg_label=0, image=img)

# --- Measure properties ---
clusters = measure.regionprops(labeled_mask)

csv_path = os.path.join(output_dir, 'grain_measurements.csv')
with open(csv_path, 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow([
        'Label', 'Area (µm^2)', 'EquivalentDiameter (µm)',
        'MajorAxisLength (µm)', 'MinorAxisLength (µm)',
        'Perimeter (µm)', 'AspectRatio', 'Circularity',
        'Solidity', 'GrainClass'
    ])

    grain_sizes = []

    for cluster_props in clusters:
        label = cluster_props.label
        area = cluster_props.area * (pixels_to_um ** 2)
        eq_diam = cluster_props.equivalent_diameter * pixels_to_um
        major = cluster_props.major_axis_length * pixels_to_um
        minor = cluster_props.minor_axis_length * pixels_to_um
        perimeter = cluster_props.perimeter * pixels_to_um
        convex_area = cluster_props.convex_area * (pixels_to_um ** 2)

        aspect_ratio = major / minor if minor > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        solidity = area / convex_area if convex_area > 0 else 0

        # Classification (Wentworth scale)
        if eq_diam < 62.5:
            grain_class = "Silt/Clay"
        elif eq_diam < 125:
            grain_class = "Very Fine Sand"
        elif eq_diam < 250:
            grain_class = "Fine Sand"
        elif eq_diam < 500:
            grain_class = "Medium Sand"
        elif eq_diam < 1000:
            grain_class = "Coarse Sand"
        else:
            grain_class = "Very Coarse Sand / Gravel"

        writer.writerow([
            label, area, eq_diam, major, minor, perimeter,
            aspect_ratio, circularity, solidity, grain_class
        ])

        grain_sizes.append(eq_diam)

# --- Visualization ---
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Labeled Grains (CNN)")
plt.imshow(img2)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grain_labels_cnn.png"), dpi=300)
plt.show()

# --- Histogram ---
plt.figure(figsize=(6, 4))
plt.hist(grain_sizes, bins=15, color='steelblue', edgecolor='black')
plt.title("Grain Size Distribution")
plt.xlabel("Equivalent Diameter (µm)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "grain_size_distribution_cnn.png"), dpi=300)
plt.show()

print(f"✅ Results saved in folder: {output_dir}")
