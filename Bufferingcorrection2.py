import cv2
import numpy as np

def detect_frozen_regions(video_path, diff_threshold=1.0, freeze_frame_count=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    frame_diffs = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, prev_frame = cap.read()
    frame_idx = 1

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Convert both frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute mean squared error (MSE)
        diff = np.mean((curr_gray.astype(float) - prev_gray.astype(float)) ** 2)
        frame_diffs.append(diff)
        timestamps.append(frame_idx / fps)

        prev_frame = curr_frame
        frame_idx += 1

    cap.release()
    print(f"Max frame diff: {max(frame_diffs):.2f}")
    print(f"Min frame diff: {min(frame_diffs):.6f}")
    print(f"Median diff: {np.median(frame_diffs):.4f}")

    # Detect frozen chunks
    frozen_regions = []
    count = 0
    start_time = None

    for i, diff in enumerate(frame_diffs):
        if diff < diff_threshold:
            count += 1
            if count == 1:
                start_time = timestamps[i]
        else:
            if count >= freeze_frame_count:
                end_time = timestamps[i]
                frozen_regions.append((start_time, end_time))
            count = 0

    # Final check at end of video
    if count >= freeze_frame_count:
        frozen_regions.append((start_time, timestamps[-1]))

    return frozen_regions

freezes = detect_frozen_regions(
    r'Z:\Alex\Reactivations\2025-05-19\11151\11151_2025-05-19-132352-0000.mp4',
    diff_threshold=0.2,        # much more sensitive (was 1.0)
    freeze_frame_count=10      # allow shorter freezes (was 30)
)


print("Detected frozen regions:")
for start, end in freezes:
    print(f"Freeze from {start:.2f}s to {end:.2f}s ({end - start:.2f} seconds)")
