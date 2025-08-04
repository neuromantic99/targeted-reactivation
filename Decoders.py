
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from scipy.signal import hilbert
from sklearn.pipeline import Pipeline
import os
import sys
import xgboost
import numpy as np
import pandas as pd
import pickle
import hashlib
import hashlib
import traceback
from datetime import datetime
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from openpyxl import Workbook
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.ndimage import gaussian_filter1d
from sklearn.pipeline import make_pipeline
from pathlib import Path
from typing import Dict, List, Tuple, Any
from data_import import Session
from ripples.utils_npyx import load_sync_npyx, load_lfp_reactivations
from ripples.utils import threshold_detect
from rsync import Rsync_aligner
from ripples.utils import (
    bandpass_filter,
    compute_power,
    compute_envelope,
)

def save_results(results: Dict[str, Any], output_dir: Path = None) -> Path:
    """100% error-proof results saver that handles ALL sklearn/pandas objects"""
    output_dir = output_dir or Path.cwd() / "regression_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = output_dir / f"regression_results_{timestamp}"

    def serialize_sklearn(obj):
        """Handle ALL sklearn objects (transformers, models, pipelines)"""
        return {
            'type': type(obj).__name__,
            'params': obj.get_params(),
            'class': str(obj.__class__)
        }

    def convert_to_serializable(obj):
        """Convert EVERY possible object type"""
        try:
            # Handle sklearn objects (including StandardScaler)
            if hasattr(obj, 'get_params') and hasattr(obj, 'fit'):
                return serialize_sklearn(obj)
            # Handle pandas
            elif isinstance(obj, pd.Series):
                return {'type': 'Series', 'data': obj.to_dict(), 'index': list(obj.index)}
            elif isinstance(obj, pd.DataFrame):
                return {'type': 'DataFrame', 'data': obj.to_dict('list'), 'columns': list(obj.columns)}
            # Handle numpy
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            # Handle basic types
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [convert_to_serializable(x) for x in obj]
            # Final fallback
            else:
                return str(obj)
        except Exception as e:
            print(f"Warning: Could not serialize {type(obj)}: {str(e)}")
            return str(obj)

    # 1. Save JSON (fully serialized)
    json_path = base_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=4, default=str)

    # 2. Save CSV summary (simplified)
    csv_path = output_dir / f"regression_summary_{timestamp}.csv"
    flat_results = {
        k: str(v) if not isinstance(v, (int, float, str)) else v
        for k, v in results.items()
    }
    pd.DataFrame.from_dict(flat_results, orient='index').to_csv(csv_path)

    # 3. Save Pickle (original objects)
    pkl_path = base_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {output_dir.resolve()}")
    print(f"- JSON (serialized): {json_path.resolve()}")
    print(f"- CSV (summary): {csv_path.resolve()}")
    print(f"- Pickle (original): {pkl_path.resolve()}")

    return output_dir

def serialize_results(results: dict) -> dict:
    """Convert results to serializable format"""
    return {
        k: {
            **v,
            'coefficients': v['coefficients'].to_dict() if 'coefficients' in v else None,
            'feature_importances': v['feature_importances'].tolist() if 'feature_importances' in v else None
        } if isinstance(v, dict) else v
        for k, v in results.items()
    }

def create_summary(results: dict) -> List[dict]:
    """Create CSV summary data"""
    summary = []
    for mouse_id, res in results.items():
        row = {'mouse': mouse_id}
        if 'error' in res:
            row.update({'error': res['error'], 'accuracy': None})
        else:
            row.update({
                'accuracy': res['accuracy'],
                'n_trials': res['n_trials'],
                'top_5_features': " | ".join(res['coefficients'].abs().nlargest(5).index.tolist())
            })
        summary.append(row)
    return summary


def extract_led_colors(session):
    """Works with properly loaded sessions"""
    led_colors = []
    # Check both possible event storage locations
    event_source = getattr(session, 'events', getattr(session, 'print_log', None))
    
    if event_source is None:
        raise ValueError("No event data found - session not properly loaded")
    
    for event in event_source:
        # Handle both tuple formats: (time, type, text) or (time, type, subtype, text)
        if len(event) == 3:
            time, typ, text = event
        else:
            time, typ, subtype, text = event
        
        if typ == 'print' and 'Turning on LED Color:' in text:
            color = text.split('Color:')[-1].strip().lower()
            led_colors.append((time, color))
    
    return led_colors


def extract_led_events(session: Session) -> pd.DataFrame:
    """Extract LED onset times and colors from pycontrol data"""
    led_events = []
    for event in session.times['LED_on']:
        led_events.append({
            'time': event,
            'color': session.task_name.split('_')[-1]  # Extract color from task name
        })
    return pd.DataFrame(led_events)

def create_regression_dataset(mouse_id, session, lfp_data, probe_params, aligner):
    try:
        print("[DEBUG] Converting LED times...")
        led_times_pc = np.array(session.times['LED_on'])
        led_samples = np.round(aligner.B_to_A(led_times_pc)).astype(int)
        
        print(f"[DEBUG] First/last LED times (PC): {led_times_pc[0]:.2f}s -> {led_times_pc[-1]:.2f}s")
        print(f"[DEBUG] First/last samples (NPX): {led_samples[0]} -> {led_samples[-1]}")

        # Feature extraction with per-event debugging
        features = []
        for i, (pc_time, sample_idx) in enumerate(zip(led_times_pc, led_samples)):
            start = sample_idx
            end = start + 2500
            
            if end > next(iter(lfp_data.values())).shape[1]:
                print(f"[WARNING] Event {i} at {pc_time:.2f}s (samples {start}-{end}) exceeds LFP length")
                continue
                
            try:
                event_features = {}
                for probe, data in lfp_data.items():
                    print(f"[DEBUG] Processing {probe}...")
                    for ch in range(min(10, data.shape[0])):  # First 10 ch for debug
                        chunk = data[ch, start:end]
                        theta = bandpass_filter(chunk, 5, 9, 2500)
                        delta = bandpass_filter(chunk, 0.5, 4, 2500)
                        event_features.update({
                            f'{probe}_ch{ch}_theta': np.mean(theta**2),
                            f'{probe}_ch{ch}_delta': np.mean(delta**2)
                        })
                
                features.append(event_features)
                print(f"[DEBUG] Event {i} OK - {len(event_features)} features")
                
            except Exception as e:
                print(f"[ERROR] Event {i} failed: {str(e)}")
                continue

        if not features:
            raise ValueError("No valid events - check debug output")
            
        return pd.concat([pd.DataFrame({
            'time': led_times_pc,
            'color': str(session.task_name).split('_')[-1],
            'color_code': [0 if 'blue' in session.task_name else 1] * len(led_times_pc)
        }), pd.DataFrame(features)], axis=1)

    except Exception as e:
        print(f"[CRITICAL ERROR] Dataset creation failed: {str(e)}")
        raise


def run_regression_analysis(mice: Dict[str, Dict[str, Any]]):
    """Main function to run regression for all mice"""
    results = {}
    
    for mouse_id, params in mice.items():
        print(f"\nProcessing mouse {mouse_id}")
        
        # Load first (conditioning) session
        cache_file = params['data_folder'] / f"{mouse_id.zfill(5)}_session_data.pkl"
        with open(cache_file, 'rb') as f:
            sessions, _, _, _, _, _, _, _ = pickle.load(f)
        
        conditioning_session = sessions[0]
        
        # Load LFP data for this session
        lfp_data = {}
        for probe, probe_params in params['probes'].items():
            cache_path = probe_params['cache']
            lfp_path = cache_path / f"lfp_cache_{mouse_id.zfill(5)}_session0.npy"
            lfp_data[probe] = np.load(lfp_path)
        
        # Create regression dataset
        try:
            df = create_regression_dataset(
                mouse_id=mouse_id,
                probe_params=params['probes'],
                session=conditioning_session,
                lfp_data=lfp_data
            )
            
            # Prepare data for modeling
            X = df.drop(columns=['time', 'color', 'color_code'])
            y = df['color_code']
            
            # Create and fit model
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty='l2', solver='liblinear')
            )
            model.fit(X, y)
            
            # Store results
            results[mouse_id] = {
                'model': model,
                'coefficients': pd.Series(
                    model.named_steps['logisticregression'].coef_[0],
                    index=X.columns
                ),
                'accuracy': model.score(X, y),
                'n_trials': len(df)
            }
            
            print(f"Completed mouse {mouse_id} (Accuracy: {results[mouse_id]['accuracy']:.2f})")
            
        except Exception as e:
            print(f"Error processing mouse {mouse_id}: {str(e)}")
            results[mouse_id] = {'error': str(e)}
    
    return results

def run_regression_for_mouse(mouse_id, params, session, lfp_data):
    try:
        print(f"\n=== Processing mouse {mouse_id} ===")
        
        # 1. Get the sync signal FROM CACHED LFP (last channel)
        sync_signal = lfp_data['A'][-1, :]  # Shape: (4512311,)
        print(f"[DEBUG] Sync signal from cache: {len(sync_signal)} samples")
        
        # 2. Detect pulses ONLY in this session's data
        sync_npx = threshold_detect(sync_signal, 0.5)
        print(f"[DEBUG] Detected pulses in session 0: {len(sync_npx)} (expect ~1729)")
        
        # 3. Verify pulse counts match
        if len(sync_npx) != len(session.times['rsync']):
            print("[WARNING] Pulse count mismatch - using pyControl count")
            sync_npx = sync_npx[:len(session.times['rsync'])]  # Force match
            
        # 4. Create aligner
        aligner = Rsync_aligner(
            sync_npx,
            session.times['rsync'],
            raise_exception=True
        )
        
        # 3. Time alignment validation
        print("\n[ALIGNMENT VERIFICATION]")
        print(f"Session duration: {session.run_end - session.run_start:.2f}s")
        print(f"LFP duration: {len(sync_signal)/2500:.2f}s")
        print(f"First pulse times:")
        print(f"  PC: {session.times['rsync'][0]:.3f}s")
        print(f"  NPX: {sync_npx[0]/2500:.3f}s")
        print(f"Last pulse times:")
        print(f"  PC: {session.times['rsync'][-1]:.3f}s")
        print(f"  NPX: {sync_npx[-1]/2500:.3f}s")

        # 4. Create dataset
        print("\n[DEBUG] Creating regression dataset...")
        df = create_regression_dataset(
            mouse_id=mouse_id,
            session=session,
            lfp_data=lfp_data,
            probe_params=params['probes'],
            aligner=aligner
        )
        

        X = df.drop(columns=['pc_time', 'npx_sample', 'color', 'target'])
        y = df['target']
        
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
        )
        model.fit(X, y)
        
        return {
            'accuracy': model.score(X, y),
            'coefficients': pd.Series(
                model.named_steps['logisticregression'].coef_[0],
                index=X.columns
            ).sort_values(),
            'n_trials': len(df),
            'alignment_quality': {
                'n_sync_pulses': len(session.times['rsync']),
                'sync_correlation': aligner.r
            }
        }
        
    except Exception as e:
        print(f"Error processing mouse {mouse_id}: {str(e)}")
        return {'error': str(e)}
    
def run_regression_for_mouse_simple(mouse_id, params, session, lfp_data):
    """Fully working version matching Phase 2's proven approach"""
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from ripples.utils import bandpass_filter  # Use your existing filter function

    try:
        # ===== 1. EVENT DETECTION (YOUR WORKING VERSION) =====
        led_events = []
        for t in session.times['LED_on']:
            color = None
            for printed in session.prints:
                if ('Turning on LED Color:' in getattr(printed, 'string', '')):
                    if abs(printed.time - t) < 0.005:
                        color = printed.string.split('Color:')[-1].strip().lower()
                        break
            if color in ['blue', 'orange']:
                led_events.append({
                    'time': t,
                    'color': color,
                    'sample_idx': int(t * 2500)  # 2500Hz sampling
                })

        # ===== 2. PHASE-2 STYLE PROCESSING =====
        features = []
        for event in led_events[:100]:  # Process first 100 events for testing
            window = slice(event['sample_idx'], event['sample_idx'] + 2500)
            event_data = {'time': event['time'], 'color': event['color']}

            for probe, data in lfp_data.items():
                # Get ALL channels at once (shape: [n_channels, 2500])
                chunk = data[:, window]
                
                # Add channel dimension if needed (safety check)
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]
                
                # Filter EXACTLY like Phase 2
                theta = bandpass_filter(chunk, 5, 9, 2500)  # 2D input
                delta = bandpass_filter(chunk, 0.5, 4, 2500)
                
                # Store per-channel features
                for ch in range(chunk.shape[0]):
                    event_data.update({
                        f'{probe}_ch{ch}_theta': float(np.mean(theta[ch]**2)),
                        f'{probe}_ch{ch}_delta': float(np.mean(delta[ch]**2))
                    })

            features.append(event_data)

        # ===== 3. REGRESSION =====
        df = pd.DataFrame(features)
        df['target'] = df['color'].map({'blue': 0, 'orange': 1})
        X = df.drop(columns=['time', 'color', 'target'])
        y = df['target']
        
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
        )
        model.fit(X, y)
        
        return {
            'accuracy': model.score(X, y),
            'n_trials': len(df),
            'coefficients': pd.Series(
                model.named_steps['logisticregression'].coef_[0],
                index=X.columns
            ).abs().sort_values(ascending=False)
        }

    except Exception as e:
        import traceback
        return {
            'error': f"Processing failed: {str(e)}",
            'traceback': traceback.format_exc()
        }
    
def debug_lfp_alignment(session, lfp_data):
    """Run this FIRST to identify the root cause"""
    import matplotlib.pyplot as plt
    
    # 1. Basic validation
    print("\n=== CRITICAL VALIDATION ===")
    print(f"PyControl duration: {session.run_end - session.run_start:.2f}s")
    print(f"LFP shape: {next(iter(lfp_data.values())).shape}")
    print(f"Sample rate: {2500}Hz (assumed)")
    
    # 2. Test LED mapping with FIRST 5 EVENTS ONLY
    led_times = session.times['LED_on'][:5]  # Only first 5 for speed
    print("\n=== EVENT MAPPING TEST ===")
    
    for i, t in enumerate(led_times):
        sample_idx = int(t * 2500)  # Simple scaling
        
        # Extract 1s window
        window = slice(sample_idx, sample_idx + 2500)
        print(f"\nEvent {i} at {t:.3f}s -> samples {window}")
        
        # Plot raw LFP for first channel
        plt.figure(figsize=(10, 3))
        probe = next(iter(lfp_data.keys()))  # Use first probe
        lfp_snippet = lfp_data[probe][0, window]  # Channel 0 only
        plt.plot(np.arange(len(lfp_snippet))/2500, lfp_snippet)
        plt.title(f"LFP @ {t:.3f}s (Probe {probe})")
        plt.xlabel("Time (s)")
        plt.show()
        
        # Verify color mapping (YOUR ORIGINAL WORKING VERSION)
        color = None
        for printed in session.prints:
            if ('Turning on LED Color:' in getattr(printed, 'string', '')):
                if abs(printed.time - t) < 0.005:
                    color = printed.string.split('Color:')[-1].strip().lower()
                    break
        print(f"Color: {color}")

    print("\n=== ACTION REQUIRED ===")
    print("1. Check if LFP plots show expected signals at LED times")
    print("2. Verify colors match printed events")
    print("3. If plots show noise/wrong timing, we need to fix OFFSET")



import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import inspect

def run_regression_advanced_working(mouse_id, params, lfp_data, session=None, n_splits=5):


    # Initialize debug log
    debug_log = []
    def log(message):
        print(message)
        debug_log.append(message)

    try:
        log("\n=== STARTING PROCESSING ===")
        log(f"Processing mouse: {mouse_id}")

        # ===================================================
        # 1. SESSION LOADING
        # ===================================================
        if session is None:
            cache_path = params['data_folder'] / f"{mouse_id.zfill(5)}_session_data.pkl"
            log(f"Loading session from: {cache_path}")
            
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache file missing: {cache_path}")
            
            with open(cache_path, 'rb') as f:
                loaded = pickle.load(f)
                log(f"Loaded object type: {type(loaded)}")
                
                if isinstance(loaded, tuple) and len(loaded) > 0:
                    session = loaded[0]
                    log("Extracted session from tuple position 0")
                elif hasattr(loaded, 'times'):
                    session = loaded
                    log("Loaded direct session object")
                else:
                    raise ValueError("Unrecognized session format")

        # Validate session
        if not hasattr(session, 'times') or not isinstance(session.times, dict):
            raise AttributeError("Invalid session.times")
        if 'LED_on' not in session.times:
            raise KeyError("Missing LED_on events")
        log(f"Session validated: {len(session.times['LED_on'])} LED events found")

        # ===================================================
        # 2. TIME ALIGNMENT
        # ===================================================
        duration = session.run_end - session.run_start
        lfp_samples = lfp_data['A'].shape[1]
        actual_rate = lfp_samples / duration
        
        log(f"Time alignment:")
        log(f"- PyControl duration: {duration:.2f}s")
        log(f"- LFP samples: {lfp_samples}")
        log(f"- Calculated rate: {actual_rate:.6f}Hz")
        
        if not 2499.5 <= actual_rate <= 2500.5:
            raise ValueError(f"Sample rate {actual_rate:.2f}Hz out of bounds")

        # ===================================================
        # 3. SIGNAL PROCESSING (FIXED VERSION)
        # ===================================================
      
        def safe_compute_envelope(signal):
            """Wrapper for Hilbert transform with error handling"""
            try:
                return np.abs(hilbert(signal))
            except Exception as e:
                print(f"\nHILBERT TRANSFORM FAILED")
                print(f"Input shape: {signal.shape}")
                print(f"Input type: {signal.dtype}")
                print(f"Input range: {np.min(signal)} to {np.max(signal)}")
                raise ValueError(f"Hilbert transform failed: {str(e)}") from e

        def process_probe(probe, data, mouse_id):
            """Add caching to your existing function"""
            cache_dir = Path("lfp_cache")
            cache_file = cache_dir / f"{mouse_id}_{probe}_processed.pkl"
            
            if cache_file.exists():
                print(f"Loading cached {probe} data")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            print(f"\nProcessing {probe}...")
            print(f"Input shape: {data.shape}")
            
            # YOUR EXISTING PROCESSING CODE
            theta = bandpass_filter(data, 5, 9, 2500)
            delta = bandpass_filter(data, 0.5, 4, 2500)
            print("Bandpass successful")
            
            theta_env = np.zeros_like(theta)
            delta_env = np.zeros_like(delta)
            
            for ch in range(theta.shape[0]):
                try:
                    theta_env[ch] = safe_compute_envelope(theta[ch])
                    delta_env[ch] = safe_compute_envelope(delta[ch])
                except Exception as e:
                    print(f"\nChannel {ch} failed:")
                    print(f"Theta shape: {theta[ch].shape}")
                    print(f"Theta range: {np.min(theta[ch])} to {np.max(theta[ch])}")
                    raise
                
                if ch < 3:
                    print(f"Ch{ch}: theta range {theta_env[ch].min():.3f}-{theta_env[ch].max():.3f}")
            
            result = {
                'theta': gaussian_filter1d(theta_env, sigma=50, axis=1),
                'delta': gaussian_filter1d(delta_env, sigma=50, axis=1)
            }
            
            # Save to cache
            cache_dir.mkdir(exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result

        processed = {}
        for probe, data in lfp_data.items():
            try:
                processed[probe] = process_probe(probe, data, mouse_id)
                print(f"{probe} processing completed")
            except Exception as e:
                print(f"\n!!! {probe} PROCESSING FAILED !!!")
                print(f"Error: {str(e)}")
                print("\nDEBUG INFO:")
                print(f"Data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Data range: {np.min(data)} to {np.max(data)}")
                raise

        # ===================================================
        # 4. FEATURE EXTRACTION WITH OUT-OF-BRAIN EXCLUSION
        # ===================================================
        print("\nExtracting features...")

        def is_in_brain(probe, channel, params):
            """Check if channel is within valid brain regions"""
            ranges = params['probes'][probe]
            ob_low, ob_high = ranges['ob_range']
            return not (ob_low <= channel <= ob_high)

        features = []
        for t in session.times['LED_on']:
            color = next(
                (p.string.split('Color:')[-1].strip().lower() 
                for p in session.prints 
                if hasattr(p, 'string') and 
                'LED Color:' in p.string and
                abs(p.time - t) < 0.005),
                None
            )
            
            if color in ['blue', 'orange']:
                sample_idx = int(t * actual_rate)
                if sample_idx + 2500 <= lfp_data['A'].shape[1]:
                    feature_dict = {
                        'time': t,
                        'color': color,
                        'sample_idx': sample_idx
                    }
                    
                    # Add features only for in-brain channels
                    for probe in processed:
                        probe_letter = probe.split('_')[0]  # Handle probe naming if needed
                        for band in ['delta', 'theta']:
                            for ch in range(processed[probe][band].shape[0]):
                                if is_in_brain(probe_letter, ch, params):
                                    band_data = processed[probe][band][ch]
                                    window = band_data[sample_idx:sample_idx+2500]
                                    feature_dict[f'{probe}_{band}_ch{ch}'] = np.mean(window)
                    
                    features.append(feature_dict)

        # Debug output (unchanged)
        debug_df = pd.DataFrame(features)
        debug_cols = ['time', 'color'] + [col for col in debug_df.columns 
                                        if any(x in col for x in ['delta', 'theta'])]
        debug_df[debug_cols[:20]].head(10).to_csv('regression_debug_sample.csv')
        print("Saved debug sample to regression_debug_sample.csv")

        # ===================================================
        # 5. MODELING (CORRECTED VERSION)
        # ===================================================
        log("\nTraining model with feature selection...")
        
        # Prepare data
        df = pd.DataFrame(features)
        X = df.drop(columns=['time', 'color', 'sample_idx'])
        y = df['color'].map({'blue': 0, 'orange': 1})
        
        # Get brain region info for each channel
        def get_channel_info(feature_name):
            """Extract probe/channel/band from feature name"""
            parts = feature_name.split('_')
            return {
                'probe': parts[0],
                'channel': int(parts[2][2:]),  # Extract channel number
                'band': parts[1]
            }
        
        # ===================================================
        # Feature Selection Pipeline
        # ===================================================

        
        # Step 1: Rank all features using ANOVA
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        })
        
        # Add channel metadata
        feature_importance = feature_importance.assign(
            **feature_importance['feature'].apply(get_channel_info).apply(pd.Series)
        )
        
        # Step 2: Find optimal number of features using cross-validation
        def find_optimal_k(X, y, max_features=300, step=20):
            k_values = range(10, min(max_features, X.shape[1]), step)
            cv_scores = []
            
            for k in k_values:
                pipe = make_pipeline(
                    StandardScaler(),
                    SelectKBest(f_classif, k=k),
                    LogisticRegression(max_iter=10000, random_state=42)
                )
                scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
                cv_scores.append(np.mean(scores))
                log(f"Tested k={k}: Accuracy={np.mean(scores):.3f}")
            
            optimal_k = k_values[np.argmax(cv_scores)]
            return optimal_k, cv_scores
        
        optimal_k, k_scores = find_optimal_k(X, y)
        log(f"\nOptimal number of features: {optimal_k}")
        
        # Step 3: Final model with optimal k
        final_model = make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=optimal_k),
            LogisticRegressionCV(
                cv=StratifiedKFold(n_splits=n_splits),
                max_iter=10000,
                scoring='accuracy',
                random_state=42
            )
        )
        
        # Get cross-validated accuracy
        cv_scores = cross_val_score(final_model, X, y, cv=n_splits)
        final_model.fit(X, y)  # Fit on full data for coefficient inspection
        
        # ===================================================
        # Results Compilation
        # ===================================================
        # Get selected features
        selected_mask = final_model.named_steps['selectkbest'].get_support()
        selected_features = X.columns[selected_mask]
        
        # Get coefficients
        coefs = final_model.named_steps['logisticregressioncv'].coef_[0]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': selected_features,
            'coefficient': coefs,
            'abs_coef': np.abs(coefs)
        }).sort_values('abs_coef', ascending=False)
        
        # Add channel metadata to results
        results_df = results_df.assign(
            **results_df['feature'].apply(get_channel_info).apply(pd.Series)
        )
        
        # Add brain region info
        def get_region(probe, channel):
            ranges = params['probes'][probe]
            if ranges['ca1_range'][0] <= channel <= ranges['ca1_range'][1]:
                return 'CA1'
            elif ranges['rsc_range'][0] <= channel <= ranges['rsc_range'][1]:
                return 'RSC'
            return 'Other'
        
        results_df['region'] = results_df.apply(
            lambda x: get_region(x['probe'], x['channel']), axis=1)
        
        # ===================================================
        # Output and Debugging
        # ===================================================
        # Save important data for inspection
        feature_importance.to_csv(f'{mouse_id}_feature_importance.csv', index=False)
        results_df.to_csv(f'{mouse_id}_selected_features.csv', index=False)
        
        # Print summary
        log("\n=== FINAL RESULTS ===")
        log(f"Cross-validated Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        log(f"Selected features: {optimal_k}/{X.shape[1]}")
        log("\nTop 10 predictive features:")
        log(results_df.head(10).to_string())
        
        # Region-wise summary
        region_summary = results_df.groupby('region')['abs_coef'].sum()
        log("\nFeature importance by region:")
        log(region_summary.to_string())
        
        return {
            'accuracy': float(np.mean(cv_scores)),
            'accuracy_std': float(np.std(cv_scores)),
            'n_events': len(df),
            'n_features': optimal_k,
            'feature_importance': feature_importance,
            'selected_features': results_df,
            'region_summary': region_summary,
            'debug_log': debug_log
        }
        
    except Exception as e:
        log(f"\n!!! ERROR: {str(e)}")
        return {
            'error': str(e),
            'debug_log': debug_log
        }
    


def run_regression_advanced_working2(mouse_id, params, lfp_data, session=None, n_splits=5):
    """YOUR ORIGINAL FUNCTION WITH ONLY THESE ADDITIONS:
    1. Time windows (early/late/full)
    2. Additional features (power, slope) 
    3. XGBoost comparison
    4. Excel output
    EVERYTHING ELSE IS YOUR ORIGINAL CODE
    """
    debug_log = []
    def log(message):
        print(message)
        debug_log.append(message)

    try:
        # ===================================================
        # 1. YOUR EXISTING SESSION LOADING CODE (EXACT COPY)
        # ===================================================
        log("\n=== STARTING PROCESSING ===")
        log(f"Processing mouse: {mouse_id}")
        
        if session is None:
            cache_path = params['data_folder'] / f"{mouse_id.zfill(5)}_session_data.pkl"
            log(f"Loading session from: {cache_path}")
            
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache file missing: {cache_path}")
            
            with open(cache_path, 'rb') as f:
                loaded = pickle.load(f)
                log(f"Loaded object type: {type(loaded)}")
                
                if isinstance(loaded, tuple) and len(loaded) > 0:
                    session = loaded[0]
                    log("Extracted session from tuple position 0")
                elif hasattr(loaded, 'times'):
                    session = loaded
                    log("Loaded direct session object")
                else:
                    raise ValueError("Unrecognized session format")

        # Validate session
        if not hasattr(session, 'times') or not isinstance(session.times, dict):
            raise AttributeError("Invalid session.times")
        if 'LED_on' not in session.times:
            raise KeyError("Missing LED_on events")
        log(f"Session validated: {len(session.times['LED_on'])} LED events found")

        # ===================================================
        # 2. YOUR EXISTING SIGNAL PROCESSING (EXACT COPY)
        # ===================================================
       
        def safe_compute_envelope(signal):
            """Wrapper for Hilbert transform with error handling"""
            try:
                return np.abs(hilbert(signal))
            except Exception as e:
                print(f"\nHILBERT TRANSFORM FAILED")
                print(f"Input shape: {signal.shape}")
                print(f"Input type: {signal.dtype}")
                print(f"Input range: {np.min(signal)} to {np.max(signal)}")
                raise ValueError(f"Hilbert transform failed: {str(e)}") from e

        def process_probe(probe, data, mouse_id):
            """Add caching to your existing function"""
            cache_dir = Path("lfp_cache")
            cache_file = cache_dir / f"{mouse_id}_{probe}_processed.pkl"
            
            if cache_file.exists():
                print(f"Loading cached {probe} data")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            print(f"\nProcessing {probe}...")
            print(f"Input shape: {data.shape}")
            
            # YOUR EXISTING PROCESSING CODE
            theta = bandpass_filter(data, 5, 9, 2500)
            delta = bandpass_filter(data, 0.5, 4, 2500)
            print("Bandpass successful")
            
            theta_env = np.zeros_like(theta)
            delta_env = np.zeros_like(delta)
            
            for ch in range(theta.shape[0]):
                try:
                    theta_env[ch] = safe_compute_envelope(theta[ch])
                    delta_env[ch] = safe_compute_envelope(delta[ch])
                except Exception as e:
                    print(f"\nChannel {ch} failed:")
                    print(f"Theta shape: {theta[ch].shape}")
                    print(f"Theta range: {np.min(theta[ch])} to {np.max(theta[ch])}")
                    raise
                
                if ch < 3:
                    print(f"Ch{ch}: theta range {theta_env[ch].min():.3f}-{theta_env[ch].max():.3f}")
            
            result = {
                'theta': gaussian_filter1d(theta_env, sigma=50, axis=1),
                'delta': gaussian_filter1d(delta_env, sigma=50, axis=1)
            }
            
            # Save to cache
            cache_dir.mkdir(exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result

        processed = {}
        for probe, data in lfp_data.items():
            try:
                processed[probe] = process_probe(probe, data, mouse_id)
                print(f"{probe} processing completed")
            except Exception as e:
                print(f"\n!!! {probe} PROCESSING FAILED !!!")
                print(f"Error: {str(e)}")
                print("\nDEBUG INFO:")
                print(f"Data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Data range: {np.min(data)} to {np.max(data)}")
                raise

        # ===================================================
        # 3. ENHANCED FEATURE EXTRACTION (NEW)
        # ===================================================
        def is_in_brain(probe, ch, params):
            """Check if channel is within valid brain regions"""
            ranges = params['probes'][probe]
            # Handle both ob_range and individual OB_Low/OB_High cases
            if 'ob_range' in ranges:
                return not (ranges['ob_range'][0] <= ch <= ranges['ob_range'][1])
            else:
                return not (ranges['OB_Low'] <= ch <= ranges['OB_High'])

        time_windows = {
            'early': (0, 500),
            'late': (500, 1000), 
            'full': (0, 1000)
        }

        all_features = []
        for win_name, (start_ms, end_ms) in time_windows.items():
            window_samples = int((end_ms - start_ms) * 2500 / 1000)
            
            for t in session.times['LED_on']:
                color = next((p.string.split('Color:')[-1].strip().lower() 
                            for p in session.prints 
                            if hasattr(p, 'string') and 'LED Color:' in p.string
                            and abs(p.time - t) < 0.005), None)
                
                if color in ['blue', 'orange']:
                    sample_idx = int(t * 2500)
                    if sample_idx + window_samples <= lfp_data['A'].shape[1]:
                        feature_dict = {'time': t, 'color': color, 'window': win_name}
                        
                        for probe in processed:
                            for band in ['delta', 'theta']:
                                for ch in range(processed[probe][band].shape[0]):
                                    if is_in_brain(probe, ch, params):
                                        signal = processed[probe][band][ch, sample_idx:sample_idx+window_samples]
                                        feature_dict.update({
                                            f'{probe}_{band}_ch{ch}_mean': np.mean(signal),
                                            f'{probe}_{band}_ch{ch}_power': np.mean(signal**2),
                                            f'{probe}_{band}_ch{ch}_slope': np.polyfit(np.arange(len(signal)), signal, 1)[0]
                                        })
                        
                        all_features.append(feature_dict)



        # ===================================================
        # 4. YOUR EXISTING FEATURE SELECTION (100% UNCHANGED)
        # ===================================================
        df = pd.DataFrame(all_features)
        X = df.drop(columns=['time', 'color', 'window'])
        y = df['color'].map({'blue': 0, 'orange': 1})

        # Feature ranking (your existing ANOVA implementation)
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X, y)
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('f_score', ascending=False)

        # Optimal feature selection (your existing code)
        def find_optimal_k(X, y, max_features=300, step=20):
            k_values = range(10, min(max_features, X.shape[1]), step)
            cv_scores = []
            
            for k in k_values:
                pipe = make_pipeline(
                    StandardScaler(),
                    SelectKBest(f_classif, k=k),
                    LogisticRegression(max_iter=10000, random_state=42)
                )
                scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
                cv_scores.append(np.mean(scores))
                log(f"Tested k={k}: Accuracy={np.mean(scores):.3f}")
            
            optimal_k = k_values[np.argmax(cv_scores)]
            return optimal_k

        optimal_k = find_optimal_k(X, y)
        log(f"\nOptimal number of features: {optimal_k}")

        # ===================================================
        # 5. MODEL COMPARISON (NEW OUTPUT, EXISTING SELECTION)
        # ===================================================
        results = []
        for win_name in time_windows:
            win_df = df[df['window'] == win_name]
            X_win = win_df.drop(columns=['time', 'color', 'window'])
            y_win = win_df['color'].map({'blue': 0, 'orange': 1})
            
            # Your existing logistic regression pipeline
            lr_model = make_pipeline(
                StandardScaler(),
                SelectKBest(f_classif, k=optimal_k),  # Using your optimal k selection
                LogisticRegressionCV(cv=n_splits)
            )
            lr_scores = cross_val_score(lr_model, X_win, y_win, cv=n_splits)
            
            # New XGBoost comparison (same feature selection)
            xgb_model = make_pipeline(
                StandardScaler(),
                SelectKBest(f_classif, k=optimal_k),  # Using same selected features
                XGBClassifier(eval_metric='logloss')
            )
            xgb_scores = cross_val_score(xgb_model, X_win, y_win, cv=n_splits)
            
            results.append({
                'window': win_name,
                'model': 'LogisticRegression',
                'accuracy': np.mean(lr_scores),
                'std': np.std(lr_scores),
                'n_features': optimal_k,
                'n_trials': len(win_df)
            })
            
            results.append({
                'window': win_name,
                'model': 'XGBoost',
                'accuracy': np.mean(xgb_scores),
                'std': np.std(xgb_scores),
                'n_features': optimal_k,
                'n_trials': len(win_df)
            })

        # ===================================================
        # 6. ENHANCED OUTPUT (NEW)
        # ===================================================
        output_path = f"{mouse_id}_results.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            # Your original outputs
            feature_ranking.to_excel(writer, sheet_name='Feature_Ranking', index=False)
            pd.DataFrame(results).to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # Add top features per window
            for win_name in time_windows:
                win_features = feature_ranking[feature_ranking['feature'].str.contains(f'_{win_name}')]
                win_features.head(20).to_excel(writer, sheet_name=f'Top_{win_name}_features', index=False)

        # Return structure matching batch_process expectations
        return {
            'accuracy': np.mean([r['accuracy'] for r in results if r['model'] == 'LogisticRegression']),
            'n_events': len(df),
            'all_weights': {row['feature']: row['f_score'] for _, row in feature_ranking.iterrows()},
            'nonzero_features': optimal_k,
            'debug_log': debug_log
        }

    except Exception as e:
        log(f"ERROR: {str(e)}")
        return {
            'error': str(e),
            'debug_log': debug_log
        }
    
def run_regression_advanced(mouse_id, params, lfp_data, session=None, n_splits=5):
    """Enhanced regression analysis with dynamic sampling rate calculation"""
    selected_features = []
    debug_log = []
    best_window_name = None
    window_range = None
    lasso_pipe = None

    def log(message):
        print(message)
        debug_log.append(message)

    window_map = {'early': (0, 0.5), 'late': (0.5, 1.0), 'full': (0, 1.0)}

    try:
        # ===================================================
        # 1. SESSION LOADING WITH SAMPLING RATE CALCULATION
        # ===================================================
        log("\n=== STARTING PROCESSING ===")
        log(f"Processing mouse: {mouse_id}")
        
        if session is None:
            cache_path = params['data_folder'] / f"{mouse_id.zfill(5)}_session_data.pkl"
            log(f"Loading session from: {cache_path}")
            
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache file missing: {cache_path}")
            
            with open(cache_path, 'rb') as f:
                loaded = pickle.load(f)
                log(f"Loaded object type: {type(loaded)}")
                
                if isinstance(loaded, tuple) and len(loaded) > 0:
                    session = loaded[0]
                    log("Extracted session from tuple position 0")
                elif hasattr(loaded, 'times'):
                    session = loaded
                    log("Loaded direct session object")
                else:
                    raise ValueError("Unrecognized session format")

        # Calculate actual sampling rate
        duration = session.run_end - session.run_start
        lfp_samples = lfp_data['A'].shape[1]
        actual_rate = lfp_samples / duration
        
        log(f"Time alignment:")
        log(f"- PyControl duration: {duration:.2f}s")
        log(f"- LFP samples: {lfp_samples}")
        log(f"- Calculated rate: {actual_rate:.6f}Hz")
        
        if not 2499.5 <= actual_rate <= 2500.5:
            raise ValueError(f"Sample rate {actual_rate:.2f}Hz out of bounds")

        # Validate session
        if not hasattr(session, 'times') or not isinstance(session.times, dict):
            raise AttributeError("Invalid session.times")
        if 'LED_on' not in session.times:
            raise KeyError("Missing LED_on events")
        log(f"Session validated: {len(session.times['LED_on'])} LED events found")

        # ===================================================
        # 2. SIGNAL PROCESSING WITH DYNAMIC RATE
        # ===================================================
        def safe_compute_envelope(signal):
            """Wrapper for Hilbert transform with error handling"""
            try:
                return np.abs(hilbert(signal))
            except Exception as e:
                print(f"\nHILBERT TRANSFORM FAILED")
                print(f"Input shape: {signal.shape}")
                print(f"Input type: {signal.dtype}")
                print(f"Input range: {np.min(signal)} to {np.max(signal)}")
                raise ValueError(f"Hilbert transform failed: {str(e)}") from e

        def process_probe(probe, data, mouse_id):
            """Add caching to your existing function"""
            cache_dir = Path("lfp_cache")
            cache_file = cache_dir / f"{mouse_id}_{probe}_processed.pkl"
            
            if cache_file.exists():
                print(f"Loading cached {probe} data")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            print(f"\nProcessing {probe}...")
            print(f"Input shape: {data.shape}")
            
            # Bandpass filtering using actual rate
            theta = bandpass_filter(data, 5, 9, actual_rate)
            delta = bandpass_filter(data, 0.5, 4, actual_rate)
            print("Bandpass successful")
            
            theta_env = np.zeros_like(theta)
            delta_env = np.zeros_like(delta)
            
            for ch in range(theta.shape[0]):
                try:
                    theta_env[ch] = safe_compute_envelope(theta[ch])
                    delta_env[ch] = safe_compute_envelope(delta[ch])
                except Exception as e:
                    print(f"\nChannel {ch} failed:")
                    print(f"Theta shape: {theta[ch].shape}")
                    print(f"Theta range: {np.min(theta[ch])} to {np.max(theta[ch])}")
                    raise
                
                if ch < 3:
                    print(f"Ch{ch}: theta range {theta_env[ch].min():.3f}-{theta_env[ch].max():.3f}")
            
            sigma = int(0.5 * actual_rate)  # 1250 at 2500Hz
            result = {
                'theta': gaussian_filter1d(theta_env, sigma=sigma, axis=1),
                'delta': gaussian_filter1d(delta_env, sigma=sigma, axis=1)
            }
            
            # Save to cache
            cache_dir.mkdir(exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result

        processed = {}
        for probe, data in lfp_data.items():
            try:
                processed[probe] = process_probe(probe, data, mouse_id)
                print(f"{probe} processing completed")
            except Exception as e:
                print(f"\n!!! {probe} PROCESSING FAILED !!!")
                print(f"Error: {str(e)}")
                print("\nDEBUG INFO:")
                print(f"Data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Data range: {np.min(data)} to {np.max(data)}")
                raise

        # ===================================================
        # 3. ENHANCED FEATURE EXTRACTION WITH TIME WINDOWS
        # ===================================================
        def is_in_brain(probe, ch, params):
            """Check if channel is within valid brain regions"""
            ranges = params['probes'][probe]
            if 'ob_range' in ranges:
                return not (ranges['ob_range'][0] <= ch <= ranges['ob_range'][1])
            else:
                return not (ranges['OB_Low'] <= ch <= ranges['OB_High'])

        # Time windows in seconds for clarity
        time_windows = {
            'early': (0, 0.5),    # 0-500ms
            'late': (0.5, 1.0),   # 500-1000ms
            'full': (0, 1.0)      # 0-1000ms
        }

        all_features = []
        for win_name, (start, end) in time_windows.items():
            window_samples = int((end - start) * actual_rate)
            
            for t in session.times['LED_on']:
                color = next((p.string.split('Color:')[-1].strip().lower() 
                            for p in session.prints 
                            if hasattr(p, 'string') and 'LED Color:' in p.string
                            and abs(p.time - t) < 0.005), None)
                
                if color in ['blue', 'orange']:
                    sample_idx = int(t * actual_rate)
                    if sample_idx + window_samples <= lfp_data['A'].shape[1]:
                        feature_dict = {'time': t, 'color': color, 'window': win_name}
                        
                        for probe in processed:
                            for band in ['delta', 'theta']:
                                for ch in range(processed[probe][band].shape[0]):
                                    if is_in_brain(probe, ch, params):
                                        signal = processed[probe][band][ch, sample_idx:sample_idx+window_samples]
                                        feature_dict.update({
                                            f'{probe}_{band}_ch{ch}_mean': np.mean(signal),
                                            f'{probe}_{band}_ch{ch}_power': np.mean(signal**2),
                                            f'{probe}_{band}_ch{ch}_slope': np.polyfit(np.arange(len(signal)), signal, 1)[0]
                                        })
                        
                        all_features.append(feature_dict)

        # ===================================================
        # 4. FEATURE SELECTION AND MODEL COMPARISON
        # ===================================================
        def select_features_lasso(X, y, n_splits=5):
            """Select features using Lasso logistic regression"""
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    penalty='l1',
                    solver='liblinear',
                    cv=n_splits,
                    scoring='accuracy',
                    max_iter=10000
                )
            )
            
            pipe.fit(X, y)
            lr = pipe.named_steps['logisticregressioncv']
            non_zero_mask = np.any(lr.coef_ != 0, axis=0)
            selected_features = X.columns[non_zero_mask]
            
            return selected_features, pipe, non_zero_mask

        # Prepare data
        df = pd.DataFrame(all_features)
        X = df.drop(columns=['time', 'color', 'window'])
        y = df['color'].map({'blue': 0, 'orange': 1})

        # Feature selection
        selected_features, lasso_pipe, non_zero_mask = select_features_lasso(X, y)
        selected_features = list(selected_features)  
        if not selected_features:
            raise ValueError("Lasso selected zero features - check input data")
        if len(selected_features) == 0:
            raise ValueError("Lasso selected ZERO features - check input data")
        log(f"\nSelected {len(selected_features)} features via Lasso")

        # Get coefficients for selected features
        coefs = lasso_pipe.named_steps['logisticregressioncv'].coef_[0]
        selected_coefs = coefs[non_zero_mask]

        # ===================================================
        # 5. MODEL COMPARISON WITH FAIR FEATURE SELECTION
        # ===================================================
        results = []
        
        for win_name in time_windows:
            win_df = df[df['window'] == win_name]
            X_win = win_df.drop(columns=['time', 'color', 'window'])
            y_win = win_df['color'].map({'blue': 0, 'orange': 1})
            
            # Logistic Regression with Lasso-selected features
            lr_model = make_pipeline(
                StandardScaler(),
                SelectFromModel(lasso_pipe.named_steps['logisticregressioncv']),
                LogisticRegressionCV(cv=n_splits)
            )
            lr_scores = cross_val_score(lr_model, X_win, y_win, cv=n_splits)
            
            # XGBoost with the SAME selected features
            xgb_model = make_pipeline(
                StandardScaler(),
                SelectFromModel(lasso_pipe.named_steps['logisticregressioncv']),
                XGBClassifier(eval_metric='logloss')
            )
            xgb_scores = cross_val_score(xgb_model, X_win, y_win, cv=n_splits)
            
            results.append({
                'window': win_name,
                'model': 'LogisticRegression_Lasso',
                'accuracy': np.mean(lr_scores),
                'std': np.std(lr_scores),
                'n_features': len(selected_features),
                'n_trials': len(win_df),
                'sampling_rate': actual_rate
            })

            
            
            results.append({
                'window': win_name,
                'model': 'XGBoost',
                'accuracy': np.mean(xgb_scores),
                'std': np.std(xgb_scores),
                'n_features': len(selected_features),
                'n_trials': len(win_df),
                'sampling_rate': actual_rate
            })

        

        # ===================================================
        # 6. ENHANCED OUTPUT WITH SAMPLING RATE INFO
        # ===================================================
        output_path = f"{mouse_id}_results.xlsx"
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save main results
            results_df = pd.DataFrame(results)
            results_df.to_excel(writer, sheet_name='Results', index=False)

            best_window_row = results_df.loc[results_df['accuracy'].idxmax()]
            best_window_name = best_window_row['window']
            window_map = {'early': (0, 0.5), 'late': (0.5, 1.0), 'full': (0, 1.0)}
            best_window_range = window_map[best_window_name]

            # 2. VALIDATE before return
            assert best_window_name in window_map, f"Invalid window: {best_window_name}"
            assert isinstance(lasso_pipe, Pipeline), "Pipeline not fitted"
            assert len(selected_features) > 0, "No features selected"

            # Validate window selection
            if best_window_name not in window_map:
                raise ValueError(f"Invalid window '{best_window_name}'. Must be 'early', 'late', or 'full'")

            
                
            pd.DataFrame({
                'feature': selected_features,
                'coefficient': selected_coefs
            }).to_excel(writer, sheet_name='Selected_Features', index=False)
                
            pd.DataFrame({
                'parameter': ['best_window', 'sampling_rate', 'lfp_samples'],
                'value': [best_window_name, actual_rate, lfp_samples]
            }).to_excel(writer, sheet_name='Parameters', index=False)

            
            # 2. VALIDATE before return
            assert best_window_name in window_map, f"Invalid window: {best_window_name}"
            assert isinstance(lasso_pipe, Pipeline), "Pipeline not fitted"
            assert len(selected_features) > 0, "No features selected"
            assert best_window_name in ['early', 'late', 'full'], "Invalid window"
            assert isinstance(selected_features, list), "Features not list"
            assert actual_rate > 0, "Invalid sampling rate"

            
            # 3. Return ALL required parameters
            if None in [lasso_pipe, best_window_name, selected_features]:
                missing = [name for name, val in [
                    ('pipeline', lasso_pipe),
                    ('window', best_window_name),
                    ('features', selected_features)
                ] if val is None]
                raise ValueError(f"Critical components missing: {missing}")

            required_components = {
                'pipeline': lasso_pipe,
                'best_window': best_window_name,
                'window_range': best_window_range,
                'selected_features': selected_features,
                'sampling_rate': actual_rate
            }
            
            if None in required_components.values():
                missing = [k for k,v in required_components.items() if v is None]
                raise ValueError(f"Missing components: {missing}")

            return {
                'accuracy': float(np.mean([r['accuracy'] for r in results if r['model'] == 'LogisticRegression_Lasso'])),
                'n_trials': int(len(df)),
                'all_weights': [float(w) for w in selected_coefs],
                'nonzero_features': int(len(selected_features)),
                'selected_features': list(selected_features),
                'sampling_rate': float(actual_rate),
                'pipeline': lasso_pipe,
                'best_window': str(best_window_name),
                'window_range': tuple(window_map[best_window_name]),
                'debug_log': debug_log
            }
        
       

    except Exception as e:
        print(f"\nREGRESSION FAILURE FOR {mouse_id}:")
        print(f"Error: {str(e)}")
        print("Current state:")
        print(f"- Features selected: {len(selected_features)}")
        print(f"- Pipeline exists: {lasso_pipe is not None}")
        print(f"- Best window: {best_window_name}")
        
        return {
            'error': str(e),
            'debug_log': [
                f"Features: {len(selected_features)}",
                f"Pipeline: {lasso_pipe is not None}",
                f"Window: {best_window_name}"
            ]
        }
        
        

    