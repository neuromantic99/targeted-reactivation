from pathlib import Path
import subprocess
import json
from typing import List

import numpy as np


def get_frame_timestamps(video_path: Path) -> List | None:
    # Command to extract frame info using ffprobe
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "json",
        str(video_path),
    ]

    # Run ffprobe and capture the output
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        print("Error running ffprobe:", result.stderr)
        return []

    # Parse JSON output
    data = json.loads(result.stdout)
    timestamps = [
        float(frame["pts_time"])
        for frame in data.get("frames", [])
        if "pts_time" in frame
    ]

    print(np.max(np.diff(timestamps)))
    return timestamps


umbrella = Path("/Volumes/MarcBusche/Qichen/20250414/")

get_frame_timestamps(umbrella / "09266_2025-04-14-150325-0000.mp4")
