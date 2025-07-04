from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from nte import map_channels_to_regions
from data_import import Session
from ripples.utils_npyx import load_lfp_reactivations
from ripples.utils import (
    bandpass_filter,
    compute_power,
    compute_envelope,
)

def plot_swr_power_regions(lfp: np.ndarray, 
                         area_channel: List[str],
                         output_path: Path,
                         ripple_band: Tuple[int, int] = (120, 250),
                         plot_first_minutes: int = 5):
    """
    Generates SWR power plot with region shading using subtle pastel shades
    Args:
        lfp: LFP data (channels x samples)
        area_channel: List of brain regions for each channel
        output_path: Where to save the plot
        ripple_band: Frequency range for SWR detection
        plot_first_minutes: Duration to plot in minutes
    """
    # Common average referencing
    lfp = lfp - np.mean(lfp, axis=1, keepdims=True)
    
    # Compute SWR power for first N minutes
    samples = plot_first_minutes * 60 * 2500
    swr_power = compute_power(
        bandpass_filter(
            lfp[:, :samples], 
            ripple_band[0], 
            ripple_band[1], 
            2500, 
            order=4
        )
    )
    
    # Plotting setup
    plt.figure(figsize=(15, 6))
    plt.plot(swr_power, color='black', linewidth=1)
    
    # Auto-color regions
    unique_regions = sorted(set(area_channel), key=lambda x: area_channel.index(x))
    cmap = plt.cm.get_cmap('tab20', len(unique_regions))
    region_colors = {region: cmap(i) for i, region in enumerate(unique_regions)}
    
    # Shade regions
    current_region = area_channel[0]
    start_channel = 0
    legend_added = set()
    
    for ch, region in enumerate(area_channel[1:], 1):
        if region != current_region or ch == len(area_channel)-1:
            end_channel = ch-1 if region != current_region else ch
            color = region_colors[current_region]
            plt.axvspan(start_channel, end_channel, color=color, alpha=0.3, 
                      label=current_region if current_region not in legend_added else "")
            legend_added.add(current_region)
            start_channel = ch
            current_region = region
    
    plt.xlabel("Channel Number")
    plt.ylabel("SWR Power")
    plt.title("SWR Power by Brain Region")
    
    # Legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
             bbox_to_anchor=(1.05, 1), 
             loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    


