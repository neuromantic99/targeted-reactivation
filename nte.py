from typing import List
import numpy as np
from pathlib import Path
HERE = Path(__file__).parent

def map_channels_to_regions(AP:float, ML: float, AZ:int, elevation: int, depth: int , n_channels: int) -> List[str]:

    """Returns a list of length n-microns with the area at each micron of the probe

    (Work in progress)

    """

    import matlab.engine

 

    # Clone this: github.com/neuromantic99/neuropixels_trajectory_explorer

    # Add npy matlab to the same folder (github.com/kwikteam/npy-matlab)

    # Set the local path to the repo here:

    path_to_npte = Path("C:/Users/steve/OneDrive - University College London/Marc/Code/targeted-reactivation/neuropixels_trajectory_explorer")

 

    eng = matlab.engine.start_matlab()

    eng.cd(str(path_to_npte), nargout=0)

    probe_area_labels, probe_area_boundaries = (

        eng.neuropixels_trajectory_explorer_nogui(

            float(AP) / 1000,

            float(ML) / 1000,

            float(AZ),

            float(elevation),

            str(path_to_npte / "npy-matlab"),

            str(HERE.parent / r"Allen CCF Mouse Atlas"),

            nargout=2,

        )

    )

 

    probe_area_labels = probe_area_labels[0]

    probe_area_boundaries = np.array(probe_area_boundaries).squeeze()

 

    # Channels with depth 0 are the ones nearest the tip.

    # Channel 0 has depth 0 so start at the tip of the probe

    # (github.com/cortex-lab/neuropixels/issues/16#issuecomment-659604278)

 

    # from www.nature.com/articles/s41598-021-81127-5/figures/1

    # tip_length = 195

    # Not true but adds simplicity when manually aligning to SW

    tip_length = 0

    distance_from_tip = tip_length

 

    area_channel: List[str] = []

    for _ in range(n_channels):

 

        channel_position = int(depth - distance_from_tip)

        distance_from_tip += 10

 

        if channel_position < 0:

            area_channel.append("Outside brain")

            continue

 

        area_idx = smallest_positive_index(

            channel_position / 1000 - probe_area_boundaries

        )

        # probe_area_labels is 1 shorter than probe_area_boundaries as it

        # marks the start and end of each region. Need to subtract 1 if it's in the

        # final region

        if area_idx == len(probe_area_labels):

            area_idx -= 1

 

        area_channel.append(probe_area_labels[area_idx])

 

    return area_channel

 

def smallest_positive_index(arr: np.ndarray) -> int:

    return np.where(arr >= 0, arr, np.inf).argmin().astype(int)

