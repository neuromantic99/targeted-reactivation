from pathlib import Path


SERVER_PATH = Path("/Volumes/MarcBusche")
LOCAL_SSD = Path("/Volumes/hard_drive/targeted_reactivation")
LFP_SYNC_FOLDER = LOCAL_SSD / "lfp_syncs"


KILOSORT_UMBRELLA = SERVER_PATH / "Alex" / "Reactivations"
SERVER_CACHE_PATH = Path(
    SERVER_PATH
    / "James/caches/spontaneous_reactivation_strength_CA1_again_for_some_reason"
)
RIPPLE_PATH = Path(SERVER_PATH / "James/caches/ripples")
DETECTION_METHOD = "median"  # options 'median' or 'sd'
RIPPLE_BAND = [120, 250]
SUPRA_RIPPLE_BAND = [250, 500]
