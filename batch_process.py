import csv
import numpy as np
import pickle
import datetime
import sys
from pathlib import Path
from typing import Dict, Any, List
import tkinter as tk
from tkinter import simpledialog
from main import process_mouse, process_mouse_p2
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from predict_sounds import predict_sound_responses, load_session2_data, process_probe_lfp
from Decoders import create_regression_dataset, run_regression_for_mouse, save_results, run_regression_for_mouse_simple, debug_lfp_alignment, run_regression_advanced




def load_mouse_mappings(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load mouse data from CSV with probe information"""
    mice = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mouse_id = row['mouse_id']
            probe = row['Probe']
            
            if mouse_id not in mice:
                mice[mouse_id] = {
                    'probes': {},
                    'data_folder': Path(row['lfp_path']).parent.parent,
                    'ap': float(row['AP']),
                    'ml': float(row['ML']),
                    'az': int(row['AZ']),
                    'elevation': int(row['elevation']),
                    'depth': int(row['depth']),
                    'n_channels': int(row['n_channels']),
                }
            
            mice[mouse_id]['probes'][probe] = {
                'lfp_path': Path(row['lfp_path']),
                'cache': Path(row['Cache']),
                'ca1_range': (int(row['CA1_Low']), int(row['CA1_High'])),
                'rsc_range': (int(row['RSC_Low']), int(row['RSC_High'])),
                'ob_range': (int(row['OB_Low']), int(row['OB_High']))
            }
    return mice

def get_user_choice() -> int:
    """Allow user to select processing phase"""
    print("\nSelect processing phase:")
    print("1: Generate SWR plots (Phase 1)")
    print("2: Generate Power plots (Phase 2)")
    print("3: Regression Analysis (Phase 3)")
    
    while True:
        choice = input("Enter 1, 2 or 3: ").strip()
        if choice in ('1', '2', '3'):
            return int(choice)
        print("Invalid input. Please enter 1 or 2.")

def run_phase(mice: dict, phase: int) -> List[Any]:
    """Run the selected phase for all mice"""
    all_sessions = []
    
    for mouse_id, params in mice.items():
        if phase == 1:
            print(f"\nRunning Phase 1 for mouse: {mouse_id}")
            from main import process_mouse
            process_mouse(phase=1, **params)
        elif phase == 2:
            print(f"\nRunning Phase 2 for mouse: {mouse_id}")
            from main import process_mouse_p2
            sessions_list = process_mouse_p2(
                phase=2,
                mouse_id=mouse_id,
                probe_params=params['probes'],
                **{k: v for k, v in params.items() if k != 'probes'}
            )
            all_sessions.extend(sessions_list)
        elif phase == 3:
            print("\n=== REGRESSION ANALYSIS ===")
            regression_results = run_regression_only(mice)
            print("\n=== MODEL VALIDATION ===")
            for mouse_id, model in regression_results.items():
                print(f"{mouse_id}:")
                print(f"Keys: {list(model.keys())}")
                if 'pipeline' in model:
                    print(f"Pipeline type: {type(model['pipeline'])}")
                if 'error' in model:
                    print(f"ERROR: {model['error']}")
            print("\n=== IMMEDIATE MODEL CHECK ===")
            for mouse_id, model in regression_results.items():
                if not all(k in model for k in ['pipeline', 'best_window']):
                    print(f"❌ INVALID MODEL {mouse_id}: Missing keys")
                    print(f"    Existing keys: {list(model.keys())}")
                else:
                    print(f"✅ Valid model for {mouse_id}")
            
            # FIRST validate all regression results exist before sound prediction
            for mouse_id in mice:
                if mouse_id not in regression_results:
                    print(f"Warning: No regression results for {mouse_id}")
                    continue
                    
                model = regression_results[mouse_id]
                required_keys = ['best_window', 'window_range', 'pipeline', 
                                'selected_features', 'sampling_rate']
                missing = [k for k in required_keys if k not in model]
                if missing:
                    print(f"Invalid model for {mouse_id}: Missing {missing}")
                    regression_results[mouse_id] = None  # Mark as invalid

            
            # Modified sound prediction integration
            from predict_sounds import predict_sound_responses, load_session2_data
            
            sound_results = {}
            for mouse_id, params in mice.items():
                try:
                    # 1. Load session data with robust validation
                    cache_path = params['data_folder'] / f"{mouse_id.zfill(5)}_session_data.pkl"
                    print(f"\n[Session 2] Loading from: {cache_path}")
                    print(f"[Session 2] File exists: {cache_path.exists()}")
                    
                    if not cache_path.exists():
                        raise FileNotFoundError(f"Session 2 cache missing: {cache_path}")

                    with open(cache_path, 'rb') as f:
                        loaded = pickle.load(f)
                        if isinstance(loaded, tuple):
                            sessions = loaded[0]
                            session2 = sessions[2]
                            
                            # 1. Calculate session2-specific rate
                            duration = session2.run_end - session2.run_start
                            lfp_path = params['probes']['A']['cache'] / f"lfp_cache_{mouse_id.zfill(5)}_session2.npy"
                            lfp_samples = np.load(lfp_path).shape[1]  # Actual samples for this session
                            actual_rate = lfp_samples / duration
                            print(f"Rate used: {actual_rate}Hz")

                    # 2. Load LFP data with identical checks to Phase 2
                    
                    lfp_s2 = {}
                    for probe, probe_data in params['probes'].items():  # Uses your existing probe_params structure
                        lfp_path = probe_data['cache'] / f"lfp_cache_{mouse_id.zfill(5)}_session2.npy"
                        raw_lfp = np.load(lfp_path)
                        print(f"[DEBUG] Loading {probe} from {lfp_path}")
                        print(f"[DEBUG] {probe} LFP shape: {raw_lfp.shape}")
                        
                        # Maintain EXACT same structure as session0 processing
                        lfp_s2[probe] = process_probe_lfp(raw_lfp, actual_rate)  # Use your existing processing function
                        
                        print(f"Loaded {probe} LFP: theta shape {lfp_s2[probe]['theta'].shape}, delta shape {lfp_s2[probe]['delta'].shape}")

                    # 3. Model validation and prediction
                    model = regression_results[mouse_id]
                    predictions = predict_sound_responses(
                        model=model,
                        session2=session2,
                        lfp_data=lfp_s2,
                        actual_rate=actual_rate  # Pass the pre-calculated rate
                    )
                    
                    sound_results[mouse_id] = predictions
                    print(f"{mouse_id}: Sound Accuracy = {predictions['correct'].mean():.2f}")
                    
                except Exception as e:
                    print(f"Sound prediction failed for {mouse_id}: {str(e)}")
                    sound_results[mouse_id] = None

            return regression_results, sound_results
    
    return all_sessions

    
def run_regression_only(mice: dict) -> Dict[str, Any]:
    results = {}
    for mouse_id, params in mice.items():
        try:
            print(f"\n=== Processing mouse: {mouse_id} ===")
            
            # Debug 1: Verify paths
            cache_file = params['data_folder'] / f"{mouse_id.zfill(5)}_session_data.pkl"
            print(f"[DEBUG] Cache file path: {cache_file}")
            print(f"[DEBUG] File exists: {cache_file.exists()}")
            
            if not cache_file.exists():
                raise FileNotFoundError(f"Cache file missing: {cache_file}")

            # Debug 2: Load session data
            print("[DEBUG] Loading session data...")
            with open(cache_file, 'rb') as f:
                try:
                    loaded_data = pickle.load(f)
                    print(f"[DEBUG] Loaded data type: {type(loaded_data)}")
                    
                    if isinstance(loaded_data, tuple):
                        print(f"[DEBUG] Tuple length: {len(loaded_data)}")
                        sessions = loaded_data[0]
                        print(f"[DEBUG] Sessions type: {type(sessions)}")
                        if hasattr(sessions, '__len__'):
                            print(f"[DEBUG] Number of sessions: {len(sessions)}")
                    else:
                        sessions = loaded_data
                        
                    # Ensure event data is loaded
                    if not hasattr(sessions[0], 'events'):
                        print("[DEBUG] Loading event data for session...")
                        sessions[0].load()
                        
                except Exception as e:
                    print(f"[ERROR] Pickle load failed: {str(e)}")
                    raise

            # Debug 3: Verify session
            if not sessions:
                raise ValueError("No sessions loaded")
            print(f"[DEBUG] First session task: {sessions[0].task_name}")

            # Debug 4: Load LFP data
            print("[DEBUG] Loading LFP data...")
            lfp_data = {}
            for probe, probe_params in params['probes'].items():
                lfp_path = probe_params['cache'] / f"lfp_cache_{mouse_id.zfill(5)}_session0.npy"
                print(f"[DEBUG] Loading {probe} from {lfp_path}")
                
                if not lfp_path.exists():
                    raise FileNotFoundError(f"LFP cache missing: {lfp_path}")
                
                lfp = np.load(lfp_path)
                print(f"[DEBUG] {probe} LFP shape: {lfp.shape}")
                lfp_data[probe] = lfp

            # Run analysis
            mouse_result = run_regression_advanced(
                mouse_id=mouse_id,
                params=params,
                session=sessions[0],
                lfp_data=lfp_data,
                n_splits=5
            )
            
            # Process results (YOUR ORIGINAL VERSION)
            if 'error' in mouse_result:
                results[mouse_id] = {'error': mouse_result['error']}
            else:
                results[mouse_id] = {
                    'accuracy': float(mouse_result['accuracy']),
                    'n_trials': int(mouse_result['n_trials']),
                    'coefficients': pd.Series(
                        data=np.array(mouse_result['all_weights']),
                        index=list(mouse_result['selected_features'])
                    ),
                    'nonzero_features': int(mouse_result['nonzero_features']),
                    'pipeline': mouse_result['pipeline'],
                    'best_window': str(mouse_result['best_window']),
                    'window_range': tuple(mouse_result['window_range']),
                    'selected_features': list(mouse_result['selected_features']),
                    'sampling_rate': float(mouse_result['sampling_rate'])
                }
                
        except Exception as e:
            print(f"Error processing mouse {mouse_id}: {str(e)}")
            results[mouse_id] = {'error': str(e)}
    
    # Save and return (FIXED SYNTAX)
    save_results(results)
    return results

if __name__ == "__main__":
    try:
    
        csv_path = Path(r"Z:\Alex\Reactivations\mouse_lfp_mapping.csv")  
        mice = load_mouse_mappings(csv_path)
        
        phase = get_user_choice()  # Phase selection function
        run_phase(mice, phase)  # Processing function
        
    except FileNotFoundError:
        print("\nScript stopped because CSV file wasn't found")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")