import cv2
from ultralytics import YOLO

# Load the model
yolo = YOLO('yolov8s.pt')

# Choose the input source (0 for camera, or the path to a video file)
use_camera = True  
video_path = 'traffic.mp4' 

# Load the video capture (use camera or video file)
if use_camera:
    videoCap = cv2.VideoCapture(0)
else:
    videoCap = cv2.VideoCapture(video_path)

# Set window name and allow for resizing
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    ret, frame = videoCap.read()
    if not ret:
        print("End of video or unable to read frame")
        break

    # Resize the frame to fit in a smaller window
    frame = cv2.resize(frame, (800, 600))  # Adjust size as necessary

    results = yolo.track(frame, stream=True)

    for result in results:
        # get the class names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective color
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # show the image with the resized frame
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()
