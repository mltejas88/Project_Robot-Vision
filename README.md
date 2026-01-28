# Vision-Based Pick and Place with YOLO Pose Estimation

This repository contains the **vision and motion ROS 2 nodes** for an autonomous
pick-and-place system using a robotic manipulator.

The project includes **dataset creation, YOLOv11 pose model training, real-time
cube detection, 3D pose estimation, and prioritized pick-and-place execution**.
The pipeline is validated in simulation and tested with real-world images.

---

## Overview

This work implements an **end-to-end perception-to-manipulation pipeline** for
colored cube sorting.

The system consists of:
1. **Training a YOLOv11 pose estimation model** for cube detection and keypoint-based pose estimation
2. A **ROS 2 vision node** that performs real-time inference and publishes 3D cube pose
3. A **ROS 2 motion node** that executes safe, prioritized pick-and-place operations

---

## YOLO Pose Model Training

### Dataset Creation
- Collected **328 cube images**
- Variations in:
  - Cube position
  - Orientation
  - Lighting conditions
- Images captured to reflect real manipulation scenarios

### Annotation
- Manual annotation using **CVAT**
- Labeled:
  - Bounding boxes
  - Keypoints for pose estimation
- Classes:
  - Red
  - Yellow
  - Green
  - Cyan cubes

### Model Training
- Model: **YOLOv11m Pose**
- Training duration: **100 epochs**
- Objective:
  - Robust cube detection
  - Accurate keypoint-based pose estimation
- Model generalized well to unseen test images

### Validation Results
- Reliable cube detection across all colors
- Accurate keypoint estimation
- Strong performance on images not used during training
- Tested on real-world images beyond the training dataset

---

## Vision Node

### Functionality
The vision node performs real-time perception using the trained YOLOv11 pose model.

**Capabilities:**
- Detects colored cubes (Red, Yellow, Green, Cyan)
- Performs keypoint-based pose estimation
- Estimates 3D cube pose relative to the robot
- Applies Z-axis calibration for accurate height estimation
- Corrects depth errors using calibration parameters
- Publishes cube color and 3D pose to a ROS 2 topic

---

### Published Topics
```text
/object_detections   (custom message: cube color + 3D pose)

