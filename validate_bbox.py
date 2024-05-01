import cv2
import numpy as np

# Load xywh values from the .npy file
xywh_values = np.load('bbox_cors.npy')

# Open the video file
video_path = "trial_vids/08_31_2024_Trial_2_20fps.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('validation_vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
frame_count = 0

# Read and process each frame
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
        print(int(x-(w/2-10)*flag))
        print(int(y-(h/2-10)*flag))
        print(x, y)
        print("--")

        # Make one pixel red
        cv2.circle(frame,(int(x),int(y)), 10, (0,0,255), -1)
    
    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    frame_count += 1
    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
