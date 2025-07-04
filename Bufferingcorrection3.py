import cv2
import numpy as np

def find_duplicated_frames(video_path, diff_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Failed to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    duplicated_frames = 0
    total_frames = 0

    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("Failed to read first frame.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = np.mean((curr_gray.astype(float) - prev_gray.astype(float)) ** 2)

        if diff < diff_threshold:
            duplicated_frames += 1

        prev_gray = curr_gray
        total_frames += 1

    cap.release()

    extra_time = duplicated_frames / fps
    print(f"Total frames: {total_frames}")
    print(f"Duplicated frames: {duplicated_frames}")
    print(f"Estimated extra time due to duplicates: {extra_time:.2f} seconds")

    return duplicated_frames, total_frames, extra_time

find_duplicated_frames(
    r'Z:\Alex\Reactivations\2025-05-19\11151\11151_2025-05-19-132352-0000.mp4',
    diff_threshold=0.51  # tweak this if needed (try 0.3 to 1.0)
)
