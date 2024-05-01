import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('MouseNet.pt')

# Open the video file
video_path = "trial_vids/08_31_2024_Trial_2_20fps.mp4"
cap = cv2.VideoCapture(video_path)
device = 'cpu'

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Loop through the video frames 
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run yolov8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame 
        # cv2.imshow("Yolov8 inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    else:
        break

# Release the video capture object, the video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()