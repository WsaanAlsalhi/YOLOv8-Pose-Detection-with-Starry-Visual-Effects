# YOLOv8-Pose-Detection-with-Starry-Visual-Effects
This project utilizes YOLOv8 to detect human poses in real-time, drawing skeletons and creating visual effects using neon-colored stars. The stars appear and fall on the screen and will avoid collisions with detected human bodies.

## Features

* Real-time human pose detection using YOLOv8.
* Draw skeletons with a dynamic starry background effect.
* Neon-colored stars falling from the top of the screen.
* Interaction with detected human poses: stars avoid drawing over the human figure.

## Requirements

* Python 3.x
* OpenCV
* Ultralyitcs YOLOv8 library
* NumPy

## Installation

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install opencv-python ultralytics numpy
   ```

## Usage

Run the script:

```bash
python yolo_pose_detection_with_stars.py
```
