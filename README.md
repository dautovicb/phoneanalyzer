# Smartphone Analyzer
Using Computer Vision to analyze images of a smartphone from OLX.ba or uploaded images.
The pipeline currently utilizes an object detection model to identify the phone's orientation, packaging and specific UI parts.
<br><br>The ultimate goal of this pipeline is to fully automate phone inspection by detecting physical damage (cracks) and extracting internal device specifications via OCR. Currently, in terms of OCR, the focus is on iPhones only.

## Detection Model
The starting point of the current pipeline is an RF-DETR object detection model.<br><br>
**Supported Classes (v2):**  
- **Hardware:** phone_front, phone_back, box, case (phone back with case)
- **Software/UI Screens:** ui_battery, ui_memory, ui_memory_about

**Metrics:**
- mAP @ 0.5: 0.9474
- mAP @ 0.5-0.95: 0.8386

*Trained in <20 epochs*

### Dataset
The [dataset](detection_model/dataset) includes 1143 images (369 original source images with 70/30 split, 258 training images).<br>
Objects are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 4 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random rotation of between -11 and +11 degrees
* Random brigthness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 1.3 pixels

## Changelog & Version History
*v2.0* (Current)

Dataset: Expanded training data to >1000 images.<br>
Classes Added: Introduced 4 new classes to handle accessories and screen information (case, ui_battery, ui_memory, ui_memory_about).<br>
Improvements: Better handling of edge cases where phones are in cases, and foundational work for the upcoming OCR extraction pipeline.<br>
<img width="395" height="207" alt="W B Chart 3_30_2026, 11_45_59 AM" src="https://github.com/user-attachments/assets/589e2359-d339-4254-bc93-18aae5fea5c2" />

*v1.0*

Dataset: ~500 images.<br>
Classes: Supported basic detection of 3 classes (phone_front, phone_back, box).<br>
Status: Deprecated.<br>
<img width="395" height="207" alt="W B Chart 3_15_2026, 3_03_05 PM" src="https://github.com/user-attachments/assets/9da3b3a9-a0fa-4aec-bba3-e59db8609acf" />
