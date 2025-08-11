import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from consts import LOCAL_SSD
from reactivation_classifier import load_spiking_data


def main() -> None:

    umbrella = Path("/Volumes/MarcBusche/Alex/Reactivations")

    for root, dirs, _ in os.walk(umbrella):
        for d in dirs:
            if "kilosort4" in d:
                full_path = Path(root) / d

                mouse = full_path.parent.parent.parent.name
                imec = f"imec_{str(full_path).split('imec')[1][0]}"

                _, _, _, closest_channel = load_spiking_data(full_path)
                n_spikes_channel = np.bincount(closest_channel, minlength=384)
                np.save(
                    LOCAL_SSD / "MUA_depths" / f"{mouse}_{imec}.npy",
                    n_spikes_channel,
                )


if __name__ == "__main__":
    main()
