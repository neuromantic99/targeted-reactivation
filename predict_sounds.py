import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import sys
from pathlib import Path
from scipy.signal import hilbert
from ripples.utils import (
    bandpass_filter,
    compute_power,
    compute_envelope,
)

from scipy.ndimage import gaussian_filter1d
import pickle

def process_probe_lfp(data, actual_rate):

    
    theta = bandpass_filter(data, 5, 9, actual_rate)
    delta = bandpass_filter(data, 0.5, 4, actual_rate)
    
    theta_env = np.abs(hilbert(theta))
    delta_env = np.abs(hilbert(delta))
    
    # Your specified 500ms smoothing window
    sigma = int(0.5 * actual_rate)  # 1250 samples at 2500Hz
    return {
        'theta': gaussian_filter1d(theta_env, sigma=sigma),
        'delta': gaussian_filter1d(delta_env, sigma=sigma)
    }

def extract_sound_events(session):
    """Handles the typo in your pycontrol file"""
    events = []
    
    # Check both possible spellings
    sound_phrases = [
        "Delivering sound frequency",  # Correct spelling
        "Deliverying sound frequency"   # Your actual typo
    ]
    
    for p in session.prints:
        if hasattr(p, 'string'):
            # Check for either spelling
            sound_msg = next(
                (phrase for phrase in sound_phrases 
                 if phrase in p.string),
                None
            )
            
            if sound_msg:
                try:
                    # Robust frequency extraction
                    freq_part = p.string.split(sound_msg)[-1].strip()
                    freq = int(''.join(c for c in freq_part if c.isdigit()))
                    
                    events.append({
                        'time': p.time,
                        'frequency': freq,
                        'correct_color': 'orange' if freq == 8000 else 'blue',
                        'source': 'print'
                    })
                except (ValueError, IndexError) as e:
                    print(f"Failed to parse sound event: {p.string} | Error: {str(e)}")
    
    # Debug output if no events found
    if not events:
        print("DEBUG: All sound-related prints:")
        for p in session.prints:
            if hasattr(p, 'string') and any(
                x.lower() in p.string.lower() 
                for x in ['deliveri', 'sound', 'frequen']
            ):
                print(f"- {p.time:.3f}s: {p.string}")
    
    return pd.DataFrame(events)

def load_session2_data(mouse_id, params, actual_rate):
    """Load and process session 2 data identically to session 0"""
    lfp_data = {}
    for probe, probe_params in params['probes'].items():
        cache_path = probe_params['cache'].parent / f"lfp_cache_{mouse_id.zfill(5)}_session2.npy"
        if not cache_path.exists():
            raise FileNotFoundError(f"Session 2 LFP not found: {cache_path}")
        
        raw_lfp = np.load(cache_path)
        lfp_data[probe] = process_probe_lfp(raw_lfp, actual_rate)
    return lfp_data

def predict_sound_responses(model, session2, lfp_data, actual_rate):
    """Uses the pre-calculated session2 rate"""
    pipeline = model['pipeline']
    features = model['selected_features']
    win_start, win_end = model['window_range']
    win_samples = int((win_end - win_start) * actual_rate)  # Uses pre-calculated rate
    
    predictions = []
    for event in extract_sound_events(session2):
        try:
            features_dict = {}
            sample_idx = int(event['time'] * actual_rate)  # Uses session2's rate
            
            # Feature extraction
            for probe in ['A', 'B']:
                for ch in range(lfp_data[probe]['delta'].shape[0]):  # 384 channels
                    for band in ['delta', 'theta']:
                        window = lfp_data[probe][band][ch, sample_idx:sample_idx+win_samples]
                        
                        features_dict.update({
                            f"{probe}_{band}_ch{ch}_mean": np.mean(window),
                            f"{probe}_{band}_ch{ch}_power": np.mean(window**2),
                            f"{probe}_{band}_ch{ch}_slope": np.polyfit(np.arange(len(window)), window, 1)[0]
                        })
            
            # Prediction
            X_pred = pd.DataFrame([features_dict])[features]
            pred = pipeline.predict(X_pred)[0]
            predictions.append({
                'time': event['time'],
                'frequency': event['frequency'],
                'predicted': 'orange' if pred == 1 else 'blue',
                'correct': int(pred == (1 if event['correct_color'] == 'orange' else 0))
            })
            
        except Exception as e:
            print(f"Skipping event at {event['time']:.2f}s: {str(e)}")
    
    return pd.DataFrame(predictions)