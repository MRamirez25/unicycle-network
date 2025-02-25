import cv2

# Paths to input videos
video1_path = "/home/mariano/phd_code/unicycle_unit/state_evolution_8_ppt_crop.mp4"
video2_path = "/home/mariano/phd_code/unicycle_unit/state_evolution_1_ppt_crop.mp4"
output_path = "difference_1_8_ppt_crop.mp4"

# Open video files
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Get video properties
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))

# Define video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break  # Stop if either video ends

    # Convert frames to grayscale (optional, improves difference visibility)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Convert grayscale diff to 3-channel image
    diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

    # Write the frame to output video
    out.write(diff_colored)

    # Display the difference
    cv2.imshow('Frame Difference', diff_colored)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break  # Press 'q' to exit

# Release resources
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
