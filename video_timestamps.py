from datetime import datetime
from pathlib import Path
import subprocess
import json
from typing import List, Optional

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
    with open("result.json", "w") as f:
        f.write(result.stdout)

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

    print(len(timestamps))
    print(np.max(np.diff(timestamps)))
    return timestamps


def get_creation_time(video_path: Path) -> Optional[datetime]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format_tags=creation_time",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        print("Error getting creation_time:", result.stderr)
        return None

    data = json.loads(result.stdout)
    try:
        creation_time_str = data["format"]["tags"]["creation_time"]
        return datetime.fromisoformat(
            creation_time_str.replace("Z", "+00:00")
        )  # handle UTC 'Z'
    except (KeyError, ValueError) as e:
        print("Could not parse creation_time:", e)
        return None
