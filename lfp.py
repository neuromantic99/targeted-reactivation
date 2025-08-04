from pathlib import Path
from typing import List
from typing import Dict, Tuple, Any

import traceback
from scipy import signal

import numpy as np
import os

from data_import import Session 

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba


from ripples.utils_npyx import load_lfp_reactivations
from ripples.utils import (
    bandpass_filter,
    compute_power,
    compute_envelope,
)

from rsync import Rsync_aligner


from pathlib import Path
import numpy as np
import seaborn as sns
from nte import map_channels_to_regions

def process_region(channels, region_name):
        try:
            print(f"\nProcessing {region_name} channels {channels}")
            
            # Extract channel data
            channel_indices = list(range(channels.start, channels.stop))
            region_lfp = lfp[channel_indices, :]
            print(f"Raw data shape: {region_lfp.shape}")
            
            # Bandpass filtering
            theta = bandpass_filter(region_lfp, 5, 9, 2500)
            delta = bandpass_filter(region_lfp, 0.5, 4, 2500)
            print(f"Filtered theta shape: {theta.shape}")
            
            # Compute envelope for each channel
            theta_env = compute_envelope(theta)
            delta_env = compute_envelope(delta)
            
            # Average across channels
            if theta_env.ndim == 2:
                theta_env = np.mean(theta_env, axis=0)
            if delta_env.ndim == 2:
                delta_env = np.mean(delta_env, axis=0)
            
            print(f"Envelope shapes - theta: {theta_env.shape}, delta: {delta_env.shape}")
            
            # Smoothing with 10-second window
            window = 25000
            pad_size = window // 2
            smooth_theta = np.convolve(theta_env, np.ones(window)/window, 'same')
            smooth_delta = np.convolve(delta_env, np.ones(window)/window, 'same')
            
            # Trim edge artifacts
            smooth_theta = smooth_theta[pad_size:-pad_size]
            smooth_delta = smooth_delta[pad_size:-pad_size]
            
            # Pad to maintain original length
            smooth_theta = np.pad(smooth_theta, (pad_size, pad_size), mode='edge')
            smooth_delta = np.pad(smooth_delta, (pad_size, pad_size), mode='edge')
            
            return smooth_theta, smooth_delta
            
        except Exception as e:
            print(f"Error processing {region_name}: {str(e)}")
            traceback.print_exc()
            return np.zeros(1000), np.zeros(1000)
        
    
def calculate_robust_limits_old(data, percentile=95):
    """
    Calculate axis limits that exclude extreme outliers
    
    """
    # Flatten and remove NaN/inf values
    clean_data = data[np.isfinite(data)]
    
    if len(clean_data) == 0:
        return 0, 1  # Default if no valid data
    
    # Get robust min/max using percentiles
    lower = np.percentile(clean_data, 100-percentile)
    upper = np.percentile(clean_data, percentile)
    
    # Add small buffer
    buffer = 0.3 * (upper - lower)
    return max(0, lower - buffer), upper + buffer

def calculate_segments_old(total_samples, max_segments=4):
    segment_length = 2500 * 60 * 10  # 10 minutes in samples
    n_full_segments = total_samples // segment_length
    remainder = total_samples % segment_length
    
    # Always show final segment if >5s of data remains
    if remainder > (2500 * 5):  # More than 5 seconds
        n_segments = n_full_segments + 1
    else:
        # Merge with last full segment
        n_segments = max(1, n_full_segments)  # At least 1 segment
    
    return min(max_segments, n_segments)

def plot_ripple_power_by_channel_old(
    lfp_path: Path, 
    mouse: str, 
    aligner: Rsync_aligner,
    session: Session,
    ca1_range: tuple,
    rsc_range: tuple, 
    aligner_id: str = "default",
    area_channel: list = None
) -> None:
    task_name = "_".join(aligner_id.split("_")[1:-1])
    print(f"CA1 range: {ca1_range}")
    print(f"RSC range: {rsc_range}")

    chunk_start_seconds = 0
    chunk_end_seconds = session.run_end
    
    lfp_cache_path = Path(f"lfp_cache_{mouse}_{aligner_id}.npy")
    if not lfp_cache_path.exists():
        print(f"Loading fresh LFP data for session {aligner_id}")
        lfp = load_lfp_reactivations(
            lfp_path,
            chunk_start_seconds=chunk_start_seconds,
            chunk_end_seconds=chunk_end_seconds,
        )
        np.save(lfp_cache_path, lfp)
    else:
        print(f"Loading cached LFP data for session {aligner_id}")
        lfp = np.load(lfp_cache_path)

    RIPPLE_BAND = [120, 250]
    # Common average referencing
    lfp = lfp - np.mean(lfp, axis=1, keepdims=True)

    THETA_BAND = [5, 9]
    DELTA_BAND = [0.5, 4]
    RIPPLE_BAND = [120, 250]

    ca1_channels = range(ca1_range[0], ca1_range[1] + 1)  # +1 to make inclusive
    rsc_channels = range(rsc_range[0], rsc_range[1] + 1)

    # Add validation for channel ranges
    def validate_channels(channels, name):
        start, end = channels
        if start < 0 or end >= lfp.shape[0]:
            raise ValueError(f"{name} range {channels} exceeds LFP channels (0-{lfp.shape[0]-1})")
        return range(start, end + 1)
    
    try:
        ca1_channels = validate_channels(ca1_range, "CA1")
        rsc_channels = validate_channels(rsc_range, "RSC")
    except ValueError as e:
        print(f"Channel range error: {e}")
        return

    print(f"\n=== DEBUG INFO ===")
    print(f"LFP array shape: {lfp.shape} (channels × samples)")
    print(f"CA1 channel range: {ca1_range}")
    print(f"RSC channel range: {rsc_range}")
    print(f"Chunk start time: {chunk_start_seconds}")
    print(f"Chunk end time: {chunk_end_seconds}")
    

    # Process each region
    def process_region(channels, region_name):
        try:
            print(f"\nProcessing {region_name} channels {channels}")
            
            # Extract channel data
            channel_indices = list(range(channels.start, channels.stop))
            region_lfp = lfp[channel_indices, :]
            print(f"Raw data shape: {region_lfp.shape}")
            
            # Bandpass filtering
            theta = bandpass_filter(region_lfp, 5, 9, 2500)
            delta = bandpass_filter(region_lfp, 0.5, 4, 2500)
            print(f"Filtered theta shape: {theta.shape}")
            
            # Compute envelope for each channel
            theta_env = compute_envelope(theta)
            delta_env = compute_envelope(delta)
            
            # Average across channels
            if theta_env.ndim == 2:
                theta_env = np.mean(theta_env, axis=0)
            if delta_env.ndim == 2:
                delta_env = np.mean(delta_env, axis=0)
            
            print(f"Envelope shapes - theta: {theta_env.shape}, delta: {delta_env.shape}")
            
            # Smoothing with 10-second window. Note that the current approach trims the edges (12500 samples from each end) and then pads back to the original length using edge values
            window = 25000
            pad_size = window // 2
            smooth_theta = np.convolve(theta_env, np.ones(window)/window, 'same')
            smooth_delta = np.convolve(delta_env, np.ones(window)/window, 'same')
            
            # Trim edge artifacts
            smooth_theta = smooth_theta[pad_size:-pad_size]
            smooth_delta = smooth_delta[pad_size:-pad_size]
            
            # Pad to maintain original length
            smooth_theta = np.pad(smooth_theta, (pad_size, pad_size), mode='edge')
            smooth_delta = np.pad(smooth_delta, (pad_size, pad_size), mode='edge')
            
            return smooth_theta, smooth_delta
            
        except Exception as e:
            print(f"Error processing {region_name}: {str(e)}")
            traceback.print_exc()
            return np.zeros(1000), np.zeros(1000)
        
    # Process both regions
    ca1_theta, ca1_delta = process_region(ca1_channels, "CA1")
    rsc_theta, rsc_delta = process_region(rsc_channels, "RSC")

    # Create time axis (in seconds)
    time_axis = np.arange(lfp.shape[1]) / 2500


    # Create output directory
    os.makedirs('power_plots', exist_ok=True)

    # Calculate segmentation
    total_samples = len(time_axis)
    segment_length = 2500 * 60 * 10  # 10 minutes in samples
    n_segments = calculate_segments(len(time_axis))
    print(f"Total duration: {len(time_axis)/2500:.1f}s → {n_segments} segments")

    for seg in range(n_segments):
        start_idx = seg * segment_length
        end_idx = min((seg + 1) * segment_length, total_samples)
        # For last segment, include any remainder
        if seg == n_segments - 1:
            end_idx = total_samples  # Capture all remaining data
        
        duration = (end_idx - start_idx)/2500
        print(f"Segment {seg+1}: {start_idx/2500:.1f}s-{end_idx/2500:.1f}s ({duration:.1f}s)")

        if start_idx >= end_idx: 
            continue

        seg_duration = (end_idx - start_idx)/2500  # Convert samples to seconds
        print(f"Processing segment {seg+1}: {start_idx/2500:.1f}s to {end_idx/2500:.1f}s ({seg_duration:.1f}s)")

        if start_idx >= total_samples:
            break
        ca1_theta_lims = calculate_robust_limits(ca1_theta[start_idx:end_idx])
        ca1_delta_lims = calculate_robust_limits(ca1_delta[start_idx:end_idx])
        rsc_theta_lims = calculate_robust_limits(rsc_theta[start_idx:end_idx])
        rsc_delta_lims = calculate_robust_limits(rsc_delta[start_idx:end_idx])
        # --- CA1 Plot ---
        plot_slice = slice(start_idx, end_idx)
        fig1, (ax1, ax1_ratio) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Power traces
        ax1.plot(time_axis[plot_slice], ca1_theta[plot_slice], 'r-', label='Theta (5-9 Hz)')
        ax1.plot(time_axis[plot_slice], ca1_delta[plot_slice], 'b-', label='Delta (0.5-4 Hz)')
        ax1.set_title(f'CA1 Power (Segment {seg+1}: {time_axis[start_idx]/60:.1f}-{time_axis[end_idx-1]/60:.1f} min)')
        ax1.set_ylabel('Power (μV)')
        # ax1.set_ylim(0, 25)
        ax1.set_ylim(0, max(ca1_theta_lims[1], ca1_delta_lims[1]) * 1.2)  # 10% headroom
        ax1.legend(loc='upper left')
        xticks = np.arange(time_axis[start_idx], time_axis[end_idx-1] + 10, 10)
        ax1.set_xticks(xticks[xticks <= time_axis[-1]])  # Ensure ticks don't exceed data bounds
        ax1.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)  # Integer seconds
        
        
        # Ratio
        ratio = np.clip(ca1_theta[start_idx:end_idx]/ca1_delta[start_idx:end_idx], 0, 4)
        ax1_ratio.plot(time_axis[start_idx:end_idx], ratio, 'g-', label='θ/Δ Ratio')
        ax1_ratio.set_ylabel('Ratio')
        # ax1_ratio.set_ylim(0, 4)
        ratio_lims = calculate_robust_limits(ratio)
        # ax1_ratio.set_ylim(ratio_lims)
        ax1_ratio.set_ylim(0, ratio_lims[1] * 1.3)
        ax1_ratio.set_xlabel('Time (seconds)')
        ax1_ratio.legend(loc='upper right')
        ax1_ratio.set_xticks(xticks[xticks <= time_axis[-1]])
        ax1_ratio.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)
        
        
        plt.tight_layout()
        plt.savefig(f'power_plots/CA1_{aligner_id}_segment_{seg+1}.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- RSC Plot ---
        fig2, (ax2, ax2_ratio) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Power traces
        ax2.plot(time_axis[plot_slice], rsc_theta[plot_slice], 'r-', label='Theta (5-9 Hz)')
        ax2.plot(time_axis[plot_slice], rsc_delta[plot_slice], 'b-', label='Delta (0.5-4 Hz)')
        ax2.set_title(f'RSC Power (Segment {seg+1}: {time_axis[start_idx]/60:.1f}-{time_axis[end_idx-1]/60:.1f} min)')
        ax2.set_ylabel('Power (μV)')
        # ax2.set_ylim(0, 10)
        ax2.set_ylim(0, max(rsc_theta_lims[1], rsc_delta_lims[1]) * 1.2)
        ax2.legend(loc='upper left')
        xticks = np.arange(time_axis[start_idx], time_axis[end_idx-1] + 10, 10)
        ax2.set_xticks(xticks[xticks <= time_axis[-1]])  # Ensure ticks don't exceed data bounds
        ax2.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)  # Integer seconds
        
        
        # Ratio
        ratio = np.clip(rsc_theta[start_idx:end_idx]/rsc_delta[start_idx:end_idx], 0, 1.5)
        ax2_ratio.plot(time_axis[start_idx:end_idx], ratio, 'g-', label='θ/Δ Ratio')
        ax2_ratio.set_ylabel('Ratio')
        # ax2_ratio.set_ylim(0, 1.5)
        ratio_lims = calculate_robust_limits(ratio)
        # ax2_ratio.set_ylim(ratio_lims)
        ax2_ratio.set_ylim(0, ratio_lims[1] * 1.3)
        ax2_ratio.set_xlabel('Time (seconds)')
        ax2_ratio.legend(loc='upper right')
        ax2_ratio.set_xticks(xticks[xticks <= time_axis[-1]])
        ax2_ratio.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'power_plots/RSC_{aligner_id}_segment_{seg+1}.jpg', dpi=300, bbox_inches='tight')
        plt.close()

def plot_ripple_power_by_channel_old(
    lfp_path: Path, 
    mouse: str, 
    aligner: Rsync_aligner,
    session: Session,
    ca1_range: tuple,
    rsc_range: tuple, 
    aligner_id: str = "default",
    area_channel: list = None,
    probe: str = None  # Added probe parameter
) -> None:
    task_name = "_".join(aligner_id.split("_")[1:-1])
    print(f"CA1 range: {ca1_range}")
    print(f"RSC range: {rsc_range}")

    chunk_start_seconds = 0
    chunk_end_seconds = session.run_end
    
    lfp_cache_path = Path(f"lfp_cache_{mouse}_{aligner_id}.npy")
    if not lfp_cache_path.exists():
        print(f"Loading fresh LFP data for session {aligner_id}")
        lfp = load_lfp_reactivations(
            lfp_path,
            chunk_start_seconds=chunk_start_seconds,
            chunk_end_seconds=chunk_end_seconds,
        )
        np.save(lfp_cache_path, lfp)
    else:
        print(f"Loading cached LFP data for session {aligner_id}")
        lfp = np.load(lfp_cache_path)

    # Common average referencing
    lfp = lfp - np.mean(lfp, axis=1, keepdims=True)

    THETA_BAND = [5, 9]
    DELTA_BAND = [0.5, 4]
    RIPPLE_BAND = [120, 250]

    ca1_channels = range(ca1_range[0], ca1_range[1] + 1)  # +1 to make inclusive
    rsc_channels = range(rsc_range[0], rsc_range[1] + 1)

    # Add validation for channel ranges
    def validate_channels(channels, name):
        start, end = channels
        if start < 0 or end >= lfp.shape[0]:
            raise ValueError(f"{name} range {channels} exceeds LFP channels (0-{lfp.shape[0]-1})")
        return range(start, end + 1)
    
    try:
        ca1_channels = validate_channels(ca1_range, "CA1")
        rsc_channels = validate_channels(rsc_range, "RSC")
    except ValueError as e:
        print(f"Channel range error: {e}")
        return

    print(f"\n=== DEBUG INFO ===")
    print(f"LFP array shape: {lfp.shape} (channels × samples)")
    print(f"CA1 channel range: {ca1_range}")
    print(f"RSC channel range: {rsc_range}")
    print(f"Chunk start time: {chunk_start_seconds}")
    print(f"Chunk end time: {chunk_end_seconds}")
    

    # Process each region
    def process_region(channels, region_name):
        try:
            print(f"\nProcessing {region_name} channels {channels}")
            
            channel_indices = list(range(channels.start, channels.stop))
            region_lfp = lfp[channel_indices, :]
            print(f"Raw data shape: {region_lfp.shape}")
            
            theta = bandpass_filter(region_lfp, 5, 9, 2500)
            delta = bandpass_filter(region_lfp, 0.5, 4, 2500)
            print(f"Filtered theta shape: {theta.shape}")
            
            theta_env = compute_envelope(theta)
            delta_env = compute_envelope(delta)
            
            if theta_env.ndim == 2:
                theta_env = np.mean(theta_env, axis=0)
            if delta_env.ndim == 2:
                delta_env = np.mean(delta_env, axis=0)
            
            print(f"Envelope shapes - theta: {theta_env.shape}, delta: {delta_env.shape}")
            
            window = 25000
            pad_size = window // 2
            smooth_theta = np.convolve(theta_env, np.ones(window)/window, 'same')
            smooth_delta = np.convolve(delta_env, np.ones(window)/window, 'same')
            
            smooth_theta = smooth_theta[pad_size:-pad_size]
            smooth_delta = smooth_delta[pad_size:-pad_size]
            
            smooth_theta = np.pad(smooth_theta, (pad_size, pad_size), mode='edge')
            smooth_delta = np.pad(smooth_delta, (pad_size, pad_size), mode='edge')
            
            return smooth_theta, smooth_delta
            
        except Exception as e:
            print(f"Error processing {region_name}: {str(e)}")
            traceback.print_exc()
            return np.zeros(1000), np.zeros(1000)
        
    # Process both regions
    ca1_theta, ca1_delta = process_region(ca1_channels, "CA1")
    rsc_theta, rsc_delta = process_region(rsc_channels, "RSC")

    # Create time axis (in seconds)
    time_axis = np.arange(lfp.shape[1]) / 2500

    # Create output directory
    os.makedirs('power_plots', exist_ok=True)

    # Calculate segmentation
    total_samples = len(time_axis)
    segment_length = 2500 * 60 * 10  # 10 minutes in samples
    n_segments = calculate_segments(len(time_axis))
    print(f"Total duration: {len(time_axis)/2500:.1f}s → {n_segments} segments")

    for seg in range(n_segments):
        start_idx = seg * segment_length
        end_idx = min((seg + 1) * segment_length, total_samples)
        if seg == n_segments - 1:
            end_idx = total_samples
        
        duration = (end_idx - start_idx)/2500
        print(f"Segment {seg+1}: {start_idx/2500:.1f}s-{end_idx/2500:.1f}s ({duration:.1f}s)")

        if start_idx >= end_idx: 
            continue

        seg_duration = (end_idx - start_idx)/2500
        print(f"Processing segment {seg+1}: {start_idx/2500:.1f}s to {end_idx/2500:.1f}s ({seg_duration:.1f}s)")

        if start_idx >= total_samples:
            break

        # Create 4-panel figure (2 regions × 2 frequency bands)
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f"Mouse {mouse} - Probe {probe} - {session.task_name} (Segment {seg+1})")
        
        # Plot settings
        colors = {'theta': 'red', 'delta': 'blue'}
        regions = {
            'CA1': (ca1_theta, ca1_delta),
            'RSC': (rsc_theta, rsc_delta)
        }
        
        # Plot each region and frequency band
        for i, (region_name, (theta, delta)) in enumerate(regions.items()):
            # Theta plot
            ax = axs[i, 0]
            ax.plot(time_axis[start_idx:end_idx], theta[start_idx:end_idx], 
                   color=colors['theta'], label='Theta (5-9 Hz)')
            ax.set_title(f'{region_name} - Theta Power')
            ax.set_ylabel('Power (μV)')
            ax.grid(axis='x', alpha=0.2, linestyle='--')
            
            # Delta plot
            ax = axs[i, 1]
            ax.plot(time_axis[start_idx:end_idx], delta[start_idx:end_idx], 
                   color=colors['delta'], label='Delta (0.5-4 Hz)')
            ax.set_title(f'{region_name} - Delta Power')
            ax.set_ylabel('Power (μV)')
            ax.grid(axis='x', alpha=0.2, linestyle='--')
            
            # Add gridlines every 10 seconds
            for j in range(2):
                axs[i,j].xaxis.set_major_locator(plt.MultipleLocator(10))
                axs[i,j].set_xlabel('Time (seconds)')
                axs[i,j].legend()
        
        plt.tight_layout()
        plt.savefig(f'power_plots/{mouse}_probe{probe}_{aligner_id}_segment_{seg+1}.jpg', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_swr_with_regions(lfp, area_channel, ripple_band=[120, 250]):
    # Compute SWR power
    swr_power = compute_power(
        bandpass_filter(lfp[:, :2500*5*60], ripple_band[0], ripple_band[1], 2500, order=4))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(swr_power, color='black', linewidth=1.2, label='SWR Power')
    
    # Generate distinct colors for all regions
    unique_regions = sorted(set(area_channel), key=lambda x: area_channel.index(x))
    color_map = {region: to_rgba(f"C{i}", alpha=0.3) 
                for i, region in enumerate(unique_regions)}
    
    # Group contiguous channels by region
    current_region = area_channel[0]
    start_channel = 0
    
    for ch, region in enumerate(area_channel[1:], 1):
        if region != current_region or ch == len(area_channel)-1:
            end_channel = ch-1 if region != current_region else ch
            ax.axvspan(start_channel-0.5, end_channel+0.5, 
                      facecolor=color_map[current_region],
                      edgecolor=None)
            start_channel = ch
            current_region = region
    
    # Create legend
    legend_patches = [Patch(facecolor=color, label=region, alpha=0.3) 
                     for region, color in color_map.items()]
    ax.legend(handles=legend_patches, 
             loc='upper left',
             bbox_to_anchor=(1.05, 1),
             title='Brain Regions')
    
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('SWR Power (120-250 Hz)')
    ax.set_title('SWR Power with Automatic Region Shading')
    plt.tight_layout()
    plt.show()

def plot_ripple_power_combined_processing(sessions_list, output_dir="power_plots_combined"):
    """Suraya's request - processes all sessions with combined filtering before splitting back"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load cached data
    cached_data = []
    for lfp_path, session, ca1_range, rsc_range, aligner_id in sessions_list:
        cache_path = Path(f"lfp_cache_{aligner_id.split('_')[0]}_{aligner_id}.npy")
        if cache_path.exists():
            print(f"Loading cached data for {aligner_id}")
            cached_data.append({
                'lfp': np.load(cache_path),
                'session': session,
                'ca1_range': ca1_range,
                'rsc_range': rsc_range,
                'aligner_id': aligner_id,
                'original_length': np.load(cache_path).shape[1]  # Store original length
            })
        else:
            print(f"No cache found for {aligner_id} - skipping")
    
    if not cached_data:
        print("No cached data available")
        return

    # 2. Concatenate raw LFP data for each region
    combined_ca1_lfp = np.hstack([d['lfp'][range(d['ca1_range'][0], d['ca1_range'][1]+1), :] 
                                for d in cached_data])
    combined_rsc_lfp = np.hstack([d['lfp'][range(d['rsc_range'][0], d['rsc_range'][1]+1), :] 
                                for d in cached_data])

    # 3. Process combined data through bandpass and Hilbert
    def process_combined(lfp_data):
        """Process concatenated data through same pipeline"""
        # Bandpass filtering
        theta = bandpass_filter(lfp_data, 5, 9, 2500)
        delta = bandpass_filter(lfp_data, 0.5, 4, 2500)
        
        # Compute envelope for each channel
        theta_env = compute_envelope(theta)
        delta_env = compute_envelope(delta)
        
        # Average across channels
        if theta_env.ndim == 2:
            theta_env = np.mean(theta_env, axis=0)
        if delta_env.ndim == 2:
            delta_env = np.mean(delta_env, axis=0)
            
        return theta_env, delta_env
    
    # Process all CA1 and RSC data together
    ca1_theta_env, ca1_delta_env = process_combined(combined_ca1_lfp)
    rsc_theta_env, rsc_delta_env = process_combined(combined_rsc_lfp)

    # 4. Split back into individual sessions
    time_offsets = [0]
    for d in cached_data:
        time_offsets.append(time_offsets[-1] + d['original_length'])
    
    # 5. Process each session with its portion of the filtered data
    for i, data in enumerate(cached_data):
        start = time_offsets[i]
        end = time_offsets[i+1]
        
        # Extract this session's data
        session_ca1_theta = ca1_theta_env[start:end]
        session_ca1_delta = ca1_delta_env[start:end]
        session_rsc_theta = rsc_theta_env[start:end] 
        session_rsc_delta = rsc_delta_env[start:end]

        # Smooth with original parameters (same as plot_ripple_power_by_channel)
        window = 25000
        pad_size = window // 2
        
        smooth_ca1_theta = np.convolve(session_ca1_theta, np.ones(window)/window, 'same')
        smooth_ca1_delta = np.convolve(session_ca1_delta, np.ones(window)/window, 'same')
        smooth_rsc_theta = np.convolve(session_rsc_theta, np.ones(window)/window, 'same')
        smooth_rsc_delta = np.convolve(session_rsc_delta, np.ones(window)/window, 'same')
        
        # Trim and pad edges (same as original)
        smooth_ca1_theta = smooth_ca1_theta[pad_size:-pad_size]
        smooth_ca1_delta = smooth_ca1_delta[pad_size:-pad_size]
        smooth_rsc_theta = smooth_rsc_theta[pad_size:-pad_size]
        smooth_rsc_delta = smooth_rsc_delta[pad_size:-pad_size]
        
        smooth_ca1_theta = np.pad(smooth_ca1_theta, (pad_size, pad_size), mode='edge')
        smooth_ca1_delta = np.pad(smooth_ca1_delta, (pad_size, pad_size), mode='edge')
        smooth_rsc_theta = np.pad(smooth_rsc_theta, (pad_size, pad_size), mode='edge')
        smooth_rsc_delta = np.pad(smooth_rsc_delta, (pad_size, pad_size), mode='edge')

        # Create time axis
        time_axis = np.arange(len(smooth_ca1_theta)) / 2500
        
        # Calculate segmentation (same as original)
        total_samples = len(time_axis)
        segment_length = 2500 * 60 * 10  # 10 minutes in samples
        n_segments = int(np.ceil(total_samples / segment_length))
        
        for seg in range(n_segments):
            start_idx = seg * segment_length
            end_idx = min((seg + 1) * segment_length, total_samples)
            
            if start_idx >= end_idx:
                continue

            # Calculate robust limits (same as original)
            ca1_theta_lims = calculate_robust_limits(smooth_ca1_theta[start_idx:end_idx])
            ca1_delta_lims = calculate_robust_limits(smooth_ca1_delta[start_idx:end_idx])
            rsc_theta_lims = calculate_robust_limits(smooth_rsc_theta[start_idx:end_idx])
            rsc_delta_lims = calculate_robust_limits(smooth_rsc_delta[start_idx:end_idx])
            
            
            # --- CA1 Plot ---
            plot_slice = slice(start_idx, end_idx)
            fig1, (ax1, ax1_ratio) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Power traces
            ax1.plot(time_axis[plot_slice], smooth_ca1_theta[plot_slice], 'r-', label='Theta (5-9 Hz)')
            ax1.plot(time_axis[plot_slice], smooth_ca1_delta[plot_slice], 'b-', label='Delta (0.5-4 Hz)')
            ax1.set_title(f'CA1 Power - {data["session"]} (Segment {seg+1}: {time_axis[start_idx]/60:.1f}-{time_axis[end_idx-1]/60:.1f} min)')
            ax1.set_ylabel('Power (μV)')
            ax1.set_ylim(0, max(ca1_theta_lims[1], ca1_delta_lims[1]) * 1.2)
            ax1.legend(loc='upper left')
            
            # Set x-ticks
            xticks = np.arange(time_axis[start_idx], time_axis[end_idx-1] + 10, 10)
            ax1.set_xticks(xticks[xticks <= time_axis[-1]])
            ax1.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)
            
            # Ratio plot
            ratio = np.clip(smooth_ca1_theta[start_idx:end_idx]/smooth_ca1_delta[start_idx:end_idx], 0, 4)
            ax1_ratio.plot(time_axis[start_idx:end_idx], ratio, 'g-', label='θ/Δ Ratio')
            ax1_ratio.set_ylabel('Ratio')
            ratio_lims = calculate_robust_limits(ratio)
            ax1_ratio.set_ylim(0, ratio_lims[1] * 1.3)
            ax1_ratio.set_xlabel('Time (seconds)')
            ax1_ratio.legend(loc='upper right')
            ax1_ratio.set_xticks(xticks[xticks <= time_axis[-1]])
            ax1_ratio.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/CA1_{data['aligner_id']}_segment_{seg+1}.jpg", dpi=300, bbox_inches='tight')
            plt.close()
            
            # --- RSC Plot ---
            fig2, (ax2, ax2_ratio) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Power traces
            ax2.plot(time_axis[plot_slice], smooth_rsc_theta[plot_slice], 'r-', label='Theta (5-9 Hz)')
            ax2.plot(time_axis[plot_slice], smooth_rsc_delta[plot_slice], 'b-', label='Delta (0.5-4 Hz)')
            ax2.set_title(f'RSC Power - {data["session"]} (Segment {seg+1}: {time_axis[start_idx]/60:.1f}-{time_axis[end_idx-1]/60:.1f} min)')
            ax2.set_ylabel('Power (μV)')
            ax2.set_ylim(0, max(rsc_theta_lims[1], rsc_delta_lims[1]) * 1.2)
            ax2.legend(loc='upper left')
            
            # Set x-ticks (same as CA1 plot)
            ax2.set_xticks(xticks[xticks <= time_axis[-1]])
            ax2.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)
            
            # Ratio plot
            ratio = np.clip(smooth_rsc_theta[start_idx:end_idx]/smooth_rsc_delta[start_idx:end_idx], 0, 1.5)
            ax2_ratio.plot(time_axis[start_idx:end_idx], ratio, 'g-', label='θ/Δ Ratio')
            ax2_ratio.set_ylabel('Ratio')
            ratio_lims = calculate_robust_limits(ratio)
            ax2_ratio.set_ylim(0, ratio_lims[1] * 1.3)
            ax2_ratio.set_xlabel('Time (seconds)')
            ax2_ratio.legend(loc='upper right')
            ax2_ratio.set_xticks(xticks[xticks <= time_axis[-1]])
            ax2_ratio.set_xticklabels(xticks[xticks <= time_axis[-1]].astype(int), rotation=90, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/RSC_{data['aligner_id']}_segment_{seg+1}.jpg", dpi=300, bbox_inches='tight')
            plt.close()
                

def plot_spectrogram(lfp: np.ndarray, sampling_rate_lfp: int) -> None:
    max_freq = 550
    edges = (
        list(range(2, 10, 1))
        + list(range(10, 100, 10))
        + list(range(100, max_freq, 50))
    )

    result = []
    for idx in range(len(edges) - 1):
        start = edges[idx]
        end = edges[idx + 1]

        result.append(
            compute_power(bandpass_filter(lfp, start, end, sampling_rate_lfp, order=4))
        )

    result = np.vstack(result)
    plt.figure()
    sns.heatmap(
        result,
        square=False,
        # cmap=sns.color_palette("YlOrBr", as_cmap=True),
        cbar_kws={"label": "Log 10 power"},
    )
    plt.ylabel("frequency")
    plt.xlabel("channel")
    plt.xticks(np.arange(384)[::10], np.arange(384)[::10])
    plt.yticks(range(len(edges)), edges)
    plt.show()

def plot_ripple_power_by_channel(
    lfp_data: Dict[str, np.ndarray],
    mouse: str,
    session: Session,
    probe_params: Dict[str, Dict[str, Any]],
    aligner_id: str = "default",
    output_dir: Path = None,
    combined_plot: bool = True
) -> None:
    """Final version with all formatting improvements requested by Suraya"""
    
    def process_region(lfp, channels, region_name):
        """Process LFP data for one region"""
        try:
            region_lfp = lfp[list(channels), :]
            theta = bandpass_filter(region_lfp, 5, 9, 2500)
            delta = bandpass_filter(region_lfp, 0.5, 4, 2500)
            theta_env = compute_envelope(theta)
            delta_env = compute_envelope(delta)
            
            if theta_env.ndim == 2:
                theta_env = np.mean(theta_env, axis=0)
            if delta_env.ndim == 2:
                delta_env = np.mean(delta_env, axis=0)
            
            window = 25000
            smooth_theta = np.convolve(theta_env, np.ones(window)/window, 'same')
            smooth_delta = np.convolve(delta_env, np.ones(window)/window, 'same')
            
            pad = window//2
            return (
                np.pad(smooth_theta[pad:-pad], (pad, pad), 'edge'),
                np.pad(smooth_delta[pad:-pad], (pad, pad), 'edge')
            )
        except Exception as e:
            print(f"Error in {region_name}: {e}")
            traceback.print_exc()
            return np.zeros(lfp.shape[1]), np.zeros(lfp.shape[1])

    def calculate_robust_limits(data, percentile=95):
        """Y-axis scaling with 40% buffer"""
        clean_data = data[np.isfinite(data)]
        if len(clean_data) == 0:
            return 0, 1
        lower = np.percentile(clean_data, 100-percentile)
        upper = np.percentile(clean_data, percentile)
        buffer = 0.4 * (upper - lower)  # Increased from 30% to 40% to capture genuine spikes
        return max(0, lower - buffer), upper + buffer

    
    
    
    # Process all probes
    processed = {}
    for probe, lfp in lfp_data.items():
        lfp = lfp - np.mean(lfp, axis=1, keepdims=True)
        ranges = probe_params[probe]
        processed[probe] = {
            'CA1': process_region(lfp, range(*ranges['ca1_range']), "CA1"),
            'RSC': process_region(lfp, range(*ranges['rsc_range']), "RSC")
        }

    time_axis = np.arange(next(iter(lfp_data.values())).shape[1]) / 2500
    
    # Calculate segments
    def calculate_segments(total_samples):
        segment_length = 2500 * 60 * 10
        n_full = total_samples // segment_length
        remainder = total_samples % segment_length
        return n_full + 1 if remainder > (2500 * 5) else max(1, n_full)
    
    n_segments = calculate_segments(len(time_axis))

    for seg in range(n_segments):
        segment_length = 2500 * 60 * 10
        start_idx = seg * segment_length
        end_idx = min((seg + 1) * segment_length, len(time_axis))
        segment_duration = (end_idx - start_idx)/2500
        plot_slice = slice(start_idx, end_idx)
        
        if start_idx >= end_idx:
            continue

        # Create figure
        fig, axs = plt.subplots(4, 1, figsize=(16, 18))
        fig.suptitle(f"{mouse} - {session.task_name} (Segment {seg+1}/{n_segments})", y=0.99)
        
        # Gridline settings - VERY VISIBLEtried to make sure that they're visible without dominating the plot
        MAJOR_GRID = {
            'color': '#000000',  # Pure black
            'alpha': 1.0,       # Fully opaque
            'linestyle': '-',   # Solid line
            'linewidth': 1.2    # Thicker lines
        }
        LABEL_STYLE = {
            'rotation': 90,
            'verticalalignment': 'center',
            'fontsize': 10
        }

        # Calculate ticks
        all_10s_ticks = np.arange(
            np.ceil(time_axis[start_idx]/10)*10,
            time_axis[end_idx-1] + 10,
            10
        )
        
       
        if segment_duration <= 20:
            num_labels = min(5, int(segment_duration) + 1) 
            label_ticks = np.linspace(
                time_axis[start_idx],
                time_axis[end_idx-1],
                num_labels
            )
            label_ticks = np.unique(np.round(label_ticks))
        else:
            label_ticks = np.arange(
                np.ceil(time_axis[start_idx]/20)*20,
                time_axis[end_idx-1] + 20,
                20
            )
        
        # Ensure final point is labeled
        if len(label_ticks) == 0 or label_ticks[-1] < time_axis[end_idx-1] - 1:
            label_ticks = np.append(label_ticks, time_axis[end_idx-1])

        # Plot each subplot
        for i, (probe, region) in enumerate([('A', 'CA1'), ('A', 'RSC'), ('B', 'CA1'), ('B', 'RSC')]):
            if probe not in processed:
                continue
                
            ax = axs[i]
            theta, delta = processed[probe][region]
            
            # Plot data
            ax.plot(time_axis[plot_slice], theta[plot_slice], 'r-', label='Theta (5-9 Hz)')
            ax.plot(time_axis[plot_slice], delta[plot_slice], 'b-', label='Delta (0.5-4 Hz)')
            
            # Formatting
            ax.set_title(f"Probe {probe} - {region} Power", pad=10)
            ax.set_ylabel('Power (μV)', labelpad=10)
            ax.set_ylim(*calculate_robust_limits(np.concatenate([
                theta[plot_slice], 
                delta[plot_slice]
            ])))
            
            # Set ticks and DARK gridlines
            ax.set_xticks(all_10s_ticks)
            ax.grid(which='major', **MAJOR_GRID)
            
            # Set labels at specified points
            ax.set_xticklabels(
                [f"{int(tick)}" if tick in label_ticks else "" for tick in all_10s_ticks],
                **LABEL_STYLE
            )
            
            if i == 0:
                ax.legend(loc='upper right', framealpha=0.9)
            
            ax.set_xlim(time_axis[start_idx], time_axis[end_idx-1])
            
            if i == 3:
                ax.set_xlabel('Time (seconds)', labelpad=10)
        
        plt.tight_layout(pad=2.0, h_pad=1.5)
        output_path = output_dir / f"{mouse}_{aligner_id}_segment{seg+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
