import ffmpeg
import subprocess
import pandas as pd

def extract_frame_timestamps_ffprobe(video_path):
    cmd = [
        'ffprobe',
        '-select_streams', 'v:0',
        '-show_frames',
        '-show_entries', 'frame=pkt_pts_time,pkt_dts_time',
        '-of', 'csv',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ffprobe error:")
        print(result.stderr)
        raise RuntimeError("ffprobe failed")

    timestamps = []
    for line in result.stdout.splitlines():
        if line.startswith('frame,'):
            fields = line.strip().split(',')
            if len(fields) < 2:
                continue
            try:
                ts = float(fields[1])
                timestamps.append(ts)
            except ValueError:
                continue

    if not timestamps:
        print("No valid timestamps found!")
        return pd.DataFrame(columns=["timestamp", "delta"])

    df = pd.DataFrame({'timestamp': timestamps})
    df['delta'] = df['timestamp'].diff()
    return df


# Test on the conditioning video which seems to have the biggest problem
df = extract_frame_timestamps_ffprobe(r'Z:\Alex\Reactivations\2025-05-19\11151\11151_2025-05-19-132352-0000.mp4')

# Look for frame freezes — e.g., any gaps much larger than expected
# For 30 fps, expected delta is ~0.033s
freeze_candidates = df[df['delta'] > 0.034]  # or 0.1 if you want to be conservative

print(freeze_candidates.head(20))
print(f"Total duration from timestamps: {df['timestamp'].iloc[-1]:.2f} seconds")


