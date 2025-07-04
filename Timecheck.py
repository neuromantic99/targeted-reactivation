import subprocess

def get_video_creation_time(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet',
         '-show_entries', 'format_tags=creation_time',
         '-of', 'default=noprint_wrappers=1:nokey=0',
         video_path],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

print(get_video_creation_time(r'Z:\Alex\Reactivations\2025-05-19\11151\11151_2025-05-19-132352-0000.mp4'))

np_time = get_np_clock_time("Z:/Alex/Reactivations/2025-05-19/11151/11151_g0_t0.nidq.meta")
print(f"Neuropixels start time (RTC): {np_time}")