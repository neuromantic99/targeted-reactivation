import numpy as np

from gsheets_importer import gsheet2df
from utils import build_path_dict, get_data_paths


def main() -> None:

    path_dict = build_path_dict()
    paths_df = gsheet2df("112rq_5qilRHtYUFnFwpjDQeF4XKyTdY6qJhIwAnykN8", "Sheet1", 1)
    for mouse, kilosort_paths in path_dict.items():

        assert all(
            [
                (kilosort_path / "spike_times.npy").exists()
                for kilosort_path in kilosort_paths
            ]
        )

        data_path = kilosort_paths[0].parent.parent.parent
        _, frame_trigger_times, pycontrol_files = get_data_paths(data_path)
        conditioning_frames = np.load(frame_trigger_times[0])
        pycontrol_conditioning_time_edges = (
            conditioning_frames[0],
            conditioning_frames[-1],
        )

        resting_frames = np.load(frame_trigger_times[1])
        pycontrol_resting_time_edges = (resting_frames[0], resting_frames[-1])
        print(f"Mouse {mouse}")
        print(f"Conditioning time edges: {pycontrol_conditioning_time_edges}")


if __name__ == "__main__":
    main()
