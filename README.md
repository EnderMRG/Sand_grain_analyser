# ShoreScan

🔹 Description

This repository contains a low-cost camera-based automated sand grain analysis pipeline for beach sediment classification.
It integrates classical image processing (OpenCV + skimage) with deep learning models (CNN / U-Net) to measure and classify sand grain sizes from images.

The workflow covers:

📷 Image Acquisition → Capture beach sand images with a low-cost camera.

⚙️ Preprocessing & Segmentation → Contrast enhancement, thresholding, watershed/U-Net.

📏 Grain Feature Extraction → Compute size, diameter, aspect ratio, circularity, solidity.

🏷 Classification → Rule-based (Wentworth scale) & CNN-based grain class prediction.

🖊 Annotation Workflow → CVAT integration for dataset creation.

📊 Outputs → CSV of grain properties, segmentation masks, labeled images, and size histograms.

🔹 Features

✅ Automated sand grain segmentation and measurement

✅ Conversion from pixels → micrometers (µm)

✅ Wentworth scale classification

✅ U-Net segmentation model training pipeline

✅ CNN classifier integration for grain categories

✅ Outputs:

- CSV with grain measurements

- Labeled segmentation maps

- Grain size distribution histograms

## Tech Stack

- Python 3.10+

- penCV, scikit-image, NumPy, SciPy, Matplotlib

- Pytorch for CNN/U-Net



##  In this repo, you’ll have a complete end-to-end solution:

- Capture sand images

- Annotate with CVAT

- Train CNN/U-Net

- Run automated measurement + classification

- Export results
## Mathematical representation

**1. Image Preprocessing**

The raw image is represented as a matrix:

- ***I(x,y)∈R<sup>H×W×3</sup>***

where ***H, W*** = image height & width, and RGB channels = 3.

**Convert to grayscale:**

- ***𝐼 <sub>g</sub>(x,y)=0.299R+0.587G+0.114B***

**Apply CLAHE (contrast enhancement):**

- ***I<sub>clahe</sub>(x,y)=CLAHE(I<sub>g</sub>(x,y))***

**Gaussian blur:**

- ***I<sub>blur</sub>​(x,y)=I<sub>clahe</sub>​(x,y)∗G<sub>σ​</sub>***


**2. Segmentation**

**Thresholding (Otsu + Adaptive):**

- ***T(x,y)=T<sub>otsu​</sub>(I<sub>blur​</sub>) ∨ T<sub>adaptive<s/ub>​(I<sub>blur</sub>​)***

**Morphological filtering (remove noise):**

- ***M(x,y)=((T∘K)∙K)***

where 
***∘*** = opening, 
***∙*** = closing, 
***K*** = structuring element.

**Watershed markers (distance transform):**

- ***D(x,y)=dist(M(x,y))***

- ***M'(x,y)=Watershed(I,D)***

So final segmentation mask:

- ***S(x,y)={M'(x,y)>0}***

**3. Feature Extraction**

For each grain region 𝑅<sub>i<sub>⊂S(x,y):

**Area:**

- ***Asub>i<sub>​=∣Rsub>i<sub>​∣⋅(p[μm])2***

where 
***p*** = pixel-to-µm scaling (e.g., 12 µm/pixel).

**Equivalent Diameter:**

- ***di​=2√(A<sub>i</sub>/​​π)***

**Major/Minor Axis (from ellipse fit):**

- ***ai<sub>i</sub>=max axis(R<sub>i</sub>​),bi<sub>i</sub>=min axis(R<sub>i</sub>​)***

**Aspect Ratio:**

- 𝐴𝑅<sub>i</sub>​=𝑎<sub>i</sub>​/𝑏<sub>i</sub>​

**Circularity:**

- ***𝐶<sub>i</sub>​=4πA<sub>i</sub>​/P<sup>2</sup><sub>i</sub>***

where 
***P<sub>i</sub>​***= perimeter.

**Solidity:**

- S<sub>i</sub>​=A<sub>i</sub>​/A<sub>convex,i</sub>

**4. Classification**

Class(d<sub>i</sub>​)=  
	Silt/Clay,
	Very Fine Sand,
	Fine Sand,
	Medium Sand,
	Coarse Sand,
	Very Coarse Sand,
	Granule
	

**CNN-based Classification:**

Each segmented grain image ***R<sub>i</sub>***​ is passed to CNN:

- ***y'<sub>i</sub>​=arg max<sub>c</sub>𝑓<sub>𝜃</sub>​(R<sub>i</sub>)***

where ***𝑓<sub>𝜃</sub>*** is the CNN with parameters 𝜃, and c = class label.

**5. Final Outputs**

**Table of measurements:**

- O={(A​<sub>i</sub>,d<sub>i</sub>​,a<sub>i</sub>​,b<sub>i</sub>,P<sub>i</sub>​,AR<sub>i</sub>​,C<sub>i</sub>​,S<sub>i</sub>​,Class<sub>i</sub>​)}<sup>N</sup><sub>i=1</sub>

where ***N*** = number of grains.

**Histogram distribution:**

- ***𝐻(d)=∑<sup>N</sup><sub>i=1</sub>1{d<sub>i</sub>∈ bin (d)}***
