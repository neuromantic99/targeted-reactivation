from pathlib import Path


LOCAL_SSD = Path("/Volumes/hard_drive/targeted_reactivation")
DETECTION_METHOD = "median"  # options 'median' or 'sd'

RIPPLE_BAND = [120, 250]
SUPRA_RIPPLE_BAND = [250, 500]

KILOSORT_UMBRELLA = Path("/Volumes/MarcBusche/Alex/Reactivations")
SERVER_CACHE_PATH = Path(
    "/Volumes/MarcBusche/James/caches/spontaneous_reactivation_strength"
)

RIPPLE_PATH = Path("/Volumes/MarcBusche/James/caches/ripples")
LFP_SYNC_FOLDER = Path("/Volumes/hard_drive/targeted_reactivation/lfp_syncs")
