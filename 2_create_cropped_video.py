import cv2
import numpy as np

# Load original video
video_path = "trial_vids/08_31_2024_Trial_2_20fps.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


# Load xywh values from the .npy file
xywh_values = np.load('bbox_data/08_31_2024_Trial_2_20fps_bbox_cors.npy')

# Define output video properties
out = cv2.VideoWriter('08_31_2024_Trial_2_20fps_cropped.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0 

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the xywh values for the current frame
    current_xywh = xywh_values[frame_count]

    # Draw a rectangle if the xywh values are not [-1, -1, -1, -1]
    if np.any(current_xywh != [-1, -1, -1, -1]):
        x, y, w, h = current_xywh
        flag = 1
        cv2.rectangle(frame, (int(x-(w/2+20)*flag), int(y-(h/2+20)*flag)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Define the size of the bounding box around the target
    bbox_width = w  # Example value
    bbox_height = h  # Example value

    # Define the center of the target of interest in the original frame
    target_center_x = int(x)  # Example value
    target_center_y = int(y)  # Example value


    # Calculate the offset to shift the target to the center of the new frame
    new_frame_center_x = frame_width // 2
    new_frame_center_y = frame_height // 2
    offset_x = new_frame_center_x - target_center_x
    offset_y = new_frame_center_y - target_center_y

    
    # Create a new frame with the target centered
    new_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Black background
    x1 = int(max(0, new_frame_center_x - bbox_width // 2))
    y1 = int(max(0, new_frame_center_y - bbox_height // 2))
    x2 = int(min(frame_width, new_frame_center_x + bbox_width // 2))
    y2 = int(min(frame_height, new_frame_center_y + bbox_height // 2))

    nx1 = int(max(0, target_center_x - bbox_width // 2))
    ny1 = int(max(0, target_center_y - bbox_height // 2))
    nx2 = int(min(frame_width, target_center_x + bbox_width // 2))
    ny2 = int(min(frame_height, target_center_y + bbox_height // 2))


    new_frame[y1:y2, x1:x2] = frame[ny1:ny2, nx1:nx2]

    # Write the new frame to the output video
    out.write(new_frame)

    frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
