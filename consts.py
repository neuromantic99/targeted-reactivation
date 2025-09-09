from pathlib import Path


LOCAL_SSD = Path("/mnt/local-ssd/james/targeted-reactivation")
LFP_SYNC_FOLDER = Path("/mnt/local-ssd/james/targeted-reactivation/lfp_syncs")

SERVER_PATH = Path("/mnt/MarcBusche")

KILOSORT_UMBRELLA = SERVER_PATH / "Alex" / "Reactivations"

SERVER_CACHE_PATH = Path(SERVER_PATH / "James/caches/spontaneous_reactivation_strength")
RIPPLE_PATH = Path(SERVER_PATH / "James/caches/ripples")


DETECTION_METHOD = "median"  # options 'median' or 'sd'
RIPPLE_BAND = [120, 250]
SUPRA_RIPPLE_BAND = [250, 500]
