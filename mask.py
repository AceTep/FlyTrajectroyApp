# video out

import cv2
import numpy as np

# Load video
video_path = "/home/acetep/Zavrsni/primjer-podataka/W11813_A1_15_12_2023-10_10.mp4"
output_path = "output_video2.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Fixed window size
window_width = 1000
window_height = 800

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (if not already)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Apply thresholding to extract dark objects
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours of flies
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale back to BGR for colored output
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw contours around flies
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize output for display (keeps original for saving)
    display_frame = cv2.resize(output, (window_width, window_height))

    # Show the frame with fixed size
    cv2.imshow("Detected Flies", display_frame)
    cv2.resizeWindow("Detected Flies", window_width, window_height)

    # Write frame to output video
    out.write(output)

    # Exit on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_path}")