import subprocess
import numpy as np
from pathlib import Path  # Add this import

def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    return float(subprocess.run(cmd, capture_output=True, text=True).stdout)

def adjust_video_duration(video_path: Path, original_duration: float, 
                        target_duration: float, output_path: Path):
    """
    Adjusts video duration to match target.
    For longer LFP duration (target_duration > original_duration):
    - Slows down video (stretches duration)
    For shorter LFP duration:
    - Speeds up video (shrinks duration)
    """
    speed_factor = original_duration / target_duration
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-filter:v', f'setpts={speed_factor}*PTS',
        # Handle audio separately to maintain pitch - not relevant here but thought it might be useful in the future
        '-filter:a', f'atempo={1/speed_factor}',
        '-c:v', 'libx264',  # Explicit video codec
        '-preset', 'fast',  # Faster encoding with good quality
        '-crf', '18',       # Quality level (18-28 is a good range apparently)
        '-y',               # Overwrite output
        str(output_path)
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    
import subprocess
from pathlib import Path
import numpy as np

def adjust_video_to_lfp(video_path: Path, lfp_duration: float, output_path: Path):
    """
    Adjusts video duration to match LFP data
    """
    # First check if video has audio
    has_audio = False
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            str(video_path)
        ], capture_output=True, text=True, check=True)
        has_audio = len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        has_audio = False

    # Get original duration
    video_duration = float(subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ], capture_output=True, text=True).stdout)

    speed_factor = video_duration / lfp_duration

    print(f"\nAdjustment Details:")
    print(f"  Video Duration: {video_duration:.3f}s")
    print(f"  LFP Duration:   {lfp_duration:.3f}s")
    print(f"  Speed Factor:   {speed_factor:.6f}x")
    print(f"  Has Audio:      {'Yes' if has_audio else 'No'}")

    # Build appropriate FFmpeg command
    if has_audio:
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-filter_complex',
            f'[0:v]setpts={1/speed_factor}*PTS[v];[0:a]atempo={speed_factor}[a]',
            '-map', '[v]',
            '-map', '[a]',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-y',
            str(output_path)
        ]
    else:
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-filter:v', f'setpts={1/speed_factor}*PTS',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-an',  # No audio
            '-y',
            str(output_path)
        ]

    print("\nExecuting FFmpeg command:")
    print(' '.join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\nFFmpeg output:")
        print(result.stderr)

        # Verify output
        new_duration = float(subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(output_path)
        ], capture_output=True, text=True).stdout)

        error = new_duration - lfp_duration
        print(f"\nVerification:")
        print(f"  New Duration: {new_duration:.3f}s")
        print(f"  Target Duration: {lfp_duration:.3f}s")
        print(f"  Error: {error:.3f}s ({abs(error)/lfp_duration*100:.2f}%)")

        return abs(error) < 0.1  # Success if within 100ms

    except subprocess.CalledProcessError as e:
        print("\nFFmpeg failed:")
        print(e.stderr)
        return False
    
def adjust_and_split_video(video_file: Path, sessions: list, output_dir: Path) -> list:
    """Adjust a single video to match total session duration and split into session parts. Special code for mouse 10679"""
    # Calculate total duration from all sessions
    total_lfp_duration = sum(session.run_end - session.run_start for session in sessions)
    
    # First adjust the entire video to match total duration
    temp_adjusted = output_dir / "temp_adjusted.mp4"
    if not adjust_video_to_lfp(video_file, total_lfp_duration, temp_adjusted):
        raise ValueError("Failed to adjust the full video duration")
    
    # Now split into session parts
    output_files = []
    current_pos = 0.0
    
    for i, session in enumerate(sessions):
        session_duration = session.run_end - session.run_start
        output_file = output_dir / f"session_{i+1}_{video_file.name}"
        
        # Calculate start and duration for this segment
        start_time = current_pos
        end_time = current_pos + session_duration
        
        # Use ffmpeg to extract the segment
        cmd = [
            'ffmpeg',
            '-i', str(temp_adjusted),
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',  # Stream copy (no re-encoding)
            '-y',  # Overwrite without asking
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            output_files.append(output_file)
            current_pos = end_time
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to split video for session {i}: {str(e)}")
    
    # Clean up temporary file
    temp_adjusted.unlink()
    return output_files