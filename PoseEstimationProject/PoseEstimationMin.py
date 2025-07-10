import cv2  # Import OpenCV for video and image processing
import mediapipe as mp  # Import MediaPipe for pose estimation
import time  # Import time module to calculate FPS

# Initialize MediaPipe drawing utility
mpDraw = mp.solutions.drawing_utils
# Initialize MediaPipe pose model
mpPose = mp.solutions.pose
# Create a Pose object to process video frames
pose = mpPose.Pose()

# Open the video file for reading
cap = cv2.VideoCapture('PoseVideos/6.mp4')

# Set initial previous time to calculate FPS later
pTime = 0

# Start an infinite loop to process video frames one by one
while True:
    # Read a single frame from the video
    success, img = cap.read()

    # Resize the frame to a standard size for better viewing and processing
    img = cv2.resize(img, (640, 480))  # Resize to 640x480 resolution

    # Convert the image from BGR (OpenCV format) to RGB (MediaPipe format)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image using the pose detection model
    results = pose.process(imgRGB)

    # Check if pose landmarks were detected in the frame
    if results.pose_landmarks:
        # Draw green pose connections on the image using custom settings
        mpDraw.draw_landmarks(
            img,
            results.pose_landmarks,
            mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=None,  # No changes to landmark style
            connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            # Green connections
        )

        # Loop through each landmark detected
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # Get the height, width, and channels of the image
            h, w, c = img.shape
            # Print the landmark ID and its coordinates
            print(id, lm)
            # Convert normalized coordinates to pixel values
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Draw a small blue circle at the landmark position
            cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)

    # Get the current time
    cTime = time.time()
    # Calculate frames per second (FPS)
    fps = 1 / (cTime - pTime)
    # Update previous time with current time
    ptime = cTime

    # Display FPS on the top-left corner of the image
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # Show the frame with pose landmarks
    cv2.imshow("Image", img)

    # Wait for 1 ms before displaying the next frame
    cv2.waitKey(1)
