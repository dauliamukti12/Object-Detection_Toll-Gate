import cv2
from ultralytics import YOLO, solutions

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("toll_gate.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define the lines to monitor
lines = [
    [(165, 219), (198, 234)],
    [(74, 179), (110, 194)],
    [(122, 199), (148, 210)],
    [(27, 160), (61, 176)],
    [(210, 237), (237, 258)],
    [(241, 263), (276, 279)],
    [(296, 281), (323, 305)],
    [(488, 269), (560, 283)]
]

classes_to_count = [2, 5]  

# Video writer
video_writer = cv2.VideoWriter("toll_gate_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counters = [
    solutions.ObjectCounter(
        view_img=True,
        reg_pts=line,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
        view_in_counts= False
    ) for line in lines
]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Track objects in the current frame
    tracks = model.track(frame, persist=True, show=False, classes=classes_to_count, tracker = "botsort.yaml")

    # Draw lines on the frame
    for start_point, end_point in lines:
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Count objects that cross each line
    for counter in counters:
        frame = counter.start_counting(frame, tracks)

    # Write the annotated frame to the video output
    video_writer.write(frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()