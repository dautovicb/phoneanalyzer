# Smartphone Analyzer
Using Computer Vision to analyze images of a smartphone from OLX.ba or uploaded images.
The pipeline currently utilizes an object detection model to identify the phone's orientation, packaging and specific UI parts.
<br><br>The ultimate goal of this pipeline is to fully automate phone inspection by detecting physical damage (cracks) and extracting internal device specifications via OCR. Currently, in terms of OCR, the focus is on iPhones only.

## Detection Model
The starting point of the current pipeline is an RF-DETR object detection model.<br><br>
**Supported Classes (v2):**  
- **Hardware:** phone_front, phone_back, box, case (phone back with case)
- **Software/UI Screens:** ui_battery, ui_memory, ui_memory_about

  
