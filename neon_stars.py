from ultralytics import YOLO
import cv2
import numpy as np
import random
import math

# Function to draw a star with improved number of points
def draw_star(img, center, outer_radius, color, thickness=-1, line_type=cv2.LINE_AA):
    """
    Draws a 10-pointed star (5 outer + 5 inner).
    img: The image to draw on.
    center: The center of the star (x, y).
    outer_radius: The radius of the outer points.
    color: The color of the star (BGR).
    thickness: Line thickness; -1 means the shape is filled.
    line_type: Type of line (e.g., cv2.LINE_AA for smooth edges).
    """
    x, y = center
    num_points = 10
    inner_radius = outer_radius * 0.4  # Adjust this ratio to improve the star shape
    points = []
    for i in range(num_points):
        angle_deg = 36 * i - 90  # Start from the top
        angle = np.deg2rad(angle_deg)
        r = outer_radius if i % 2 == 0 else inner_radius
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        points.append((px, py))
    points = np.array(points, np.int32)
    if thickness == -1:
        cv2.fillPoly(img, [points], color, line_type)
    else:
        cv2.polylines(img, [points], True, color, thickness, line_type=line_type)

# Load YOLOv8-pose model
model = YOLO("yolov8n-pose.pt")

# Open camera
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List of neon colors (BGR)
neon_colors = [
    (20, 255, 57),    # Neon Green
    (147, 20, 255),   # Neon Pink
    (255, 77, 255),   # Neon Purple-ish
    (0, 255, 255)     # Neon Yellow
]

# Setup star drops (used for visual effect on black background)
num_drops = 500
star_drops = []
for i in range(num_drops):
    drop = {
        'x': random.randint(0, frame_width),
        'y': random.randint(0, frame_height),
        'speed': random.randint(1, 5),
        'color': random.choice(neon_colors)
    }
    star_drops.append(drop)

# Define pairs of points for drawing skeleton according to COCO keypoints:
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
# 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
# 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
# 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
skeleton_pairs = [
    (5, 7), (7, 9),        # Left arm
    (6, 8), (8, 10),       # Right arm
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15),    # Left leg
    (12, 14), (14, 16)     # Right leg
]

# Maximum distance between two points for drawing a line (can adjust based on frame size)
max_line_distance = 300  # in pixels

# Set up fullscreen window
window_name = "Skeleton Without Tracking Box"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the model to get keypoints
    results = model(frame, conf=0.5)  # Raise confidence threshold to reduce errors

    # Create a black background
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    boxes = []     # To define human area (for collision detection with stars)
    keypoints = [] # To store keypoints for each person

    if len(results) > 0:
        res = results[0]
        # Extract bounding boxes (we won't draw them)
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes_array = (
                res.boxes.xyxy.cpu().numpy()
                if hasattr(res.boxes.xyxy, "cpu")
                else res.boxes.xyxy
            )
            boxes = boxes_array.tolist()

        # Extract keypoints
        if hasattr(res, "keypoints") and res.keypoints is not None:
            # Use xyn to normalize coordinates to [0, 1] range with confidence
            kpts = (
                res.keypoints.xyn.cpu().numpy()
                if hasattr(res.keypoints.xyn, "cpu")
                else res.keypoints.xyn
            )
            # Convert coordinates to pixels
            kpts[..., 0] *= frame_width
            kpts[..., 1] *= frame_height
            keypoints = kpts.tolist()

    # Draw skeleton for each detected person
    for person in keypoints:
        coords_2d = []
        confidences = []
        for kp in person:
            if len(kp) == 3:
                x, y, c = kp
                coords_2d.append((x, y))
                confidences.append(c)
            else:
                x, y = kp
                coords_2d.append((x, y))
                confidences.append(1.0)

        # Draw lines with confidence filtering and distance check
        for (p1, p2) in skeleton_pairs:
            if p1 < len(coords_2d) and p2 < len(coords_2d):
                x1, y1 = coords_2d[p1]
                x2, y2 = coords_2d[p2]
                c1 = confidences[p1]
                c2 = confidences[p2]

                # Check confidence
                if c1 > 0.3 and c2 > 0.3:
                    # Calculate distance between two points
                    distance = math.hypot(x2 - x1, y2 - y1)
                    # Draw line only if the distance is reasonable
                    if distance < max_line_distance:
                        cv2.line(
                            black_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 255, 255),
                            2,
                            lineType=cv2.LINE_AA
                        )

        # Draw the head (using a circle around the nose point, index=0) if confident
        if len(coords_2d) > 0 and confidences[0] > 0.3:
            nose_x, nose_y = coords_2d[0]
            cv2.circle(black_frame, (int(nose_x), int(nose_y)), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Update and draw stars
    for drop in star_drops:
        drop['y'] += drop['speed']
        if drop['y'] > frame_height:
            drop['y'] = 0
            drop['x'] = random.randint(0, frame_width)
        
        # Check for collision with the body using the extracted boxes
        inside_human = False
        for b in boxes:
            x_min, y_min, x_max, y_max = b
            if x_min <= drop['x'] <= x_max and y_min <= drop['y'] <= y_max:
                inside_human = True
                break
        
        if not inside_human:
            # Draw the star using the draw_star function
            draw_star(
                black_frame,
                (drop['x'], drop['y']),
                outer_radius=8,       # Can adjust size here
                color=drop['color'],
                thickness=-1,         # Fill the star
                line_type=cv2.LINE_AA
            )
    
    # Display the final frame
    cv2.imshow(window_name, black_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
