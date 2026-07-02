"""
Download All Datasets
======================
Master script to download all datasets required for the EEG-BPE project.

Usage:
    python -m data_download.download_all              # Download everything
    python -m data_download.download_all --dry-run    # Show what would be downloaded
    python -m data_download.download_all --only bci_iv_2a physionet_mi  # Specific

Datasets:
    1. BCI Competition IV 2a   — Motor Imagery, 9 subjects    (via MOABB)
    2. PhysioNet MI            — Motor Imagery, 109 subjects   (via MNE)
    3. Sleep-EDF Expanded      — Sleep staging, 78 subjects    (via MNE)
    4. Mental Arithmetic       — Cognitive load, 36 subjects   (via PhysioNet)
    5. EPFL P300               — P300 speller, 8 subjects      (via MOABB)
    6. SSVEP Nakanishi         — SSVEP, 10 subjects            (via MOABB)
"""

import sys
# Force UTF-8 output so Unicode chars (─, ✓, ✗, etc.) work on Windows cp1251 consoles.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import logging
import argparse
import importlib
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parent / "download.log"),
    ],
)
logger = logging.getLogger(__name__)

# Registry of all datasets
DATASETS = {
    "bci_iv_2a": {
        "name": "BCI Competition IV 2a",
        "module": "download_bci_iv_2a",
        "function": "download_bci_iv_2a",
        "auto": True,
        "size_estimate": "~300 MB",
        "priority": "required",
    },
    "physionet_mi": {
        "name": "PhysioNet Motor Imagery",
        "module": "download_physionet_mi",
        "function": "download_physionet_mi",
        "auto": True,
        "size_estimate": "~1.5 GB",
        "priority": "required",
    },
    "sleep_edf": {
        "name": "Sleep-EDF Expanded",
        "module": "download_sleep_edf",
        "function": "download_sleep_edf",
        "auto": True,
        "size_estimate": "~5 GB",
        "priority": "required",
    },
    "mental_arithmetic": {
        "name": "EEG Mental Arithmetic",
        "module": "download_mental_arithmetic",
        "function": "download_mental_arithmetic",
        "auto": True,
        "size_estimate": "~250 MB",
        "priority": "required",
    },
    "epfl_p300": {
        "name": "EPFL P300 (MOABB)",
        "module": "download_epfl_p300",
        "function": "download_epfl_p300",
        "auto": True,
        "size_estimate": "~200 MB",
        "priority": "required",
    },
    "ssvep_nakanishi": {
        "name": "SSVEP Nakanishi 2015 (MOABB)",
        "module": "download_ssvep_nakanishi",
        "function": "download_ssvep_nakanishi",
        "auto": True,
        "size_estimate": "~15 MB",
        "priority": "required",
    },
}


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in MB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Download all datasets for EEG-BPE project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_all.py                              # Download all
  python download_all.py --only bci_iv_2a sleep_edf   # Specific datasets
  python download_all.py --dry-run                     # Preview only
        """,
    )
    parser.add_argument("--only", nargs="+", choices=list(DATASETS.keys()),
                        help="Download only these datasets")
    parser.add_argument("--skip", nargs="+", choices=list(DATASETS.keys()),
                        default=[], help="Skip these datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    # Determine which datasets to download
    if args.only:
        selected = args.only
    else:
        selected = [k for k in DATASETS.keys() if k not in args.skip]

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EEG-BPE Dataset Downloader — {len(selected)} datasets")
    print(f"{'='*60}")
    for i, key in enumerate(selected, 1):
        ds = DATASETS[key]
        auto_str = "auto" if ds["auto"] else "MANUAL"
        print(f"  {i}. [{auto_str:>6}] {ds['name']:<35} {ds['size_estimate']}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  --dry-run: No data will be downloaded.\n")
        return

    # Check dependencies
    _check_dependencies()

    # Download each dataset with overall progress bar
    results = {}
    t_total = time.time()

    overall_pbar = tqdm(
        selected,
        desc="Overall",
        unit="dataset",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} datasets [{elapsed}<{remaining}]",
        colour="white",
        position=0,
    )
    for key in overall_pbar:
        ds = DATASETS[key]
        overall_pbar.set_postfix_str(ds["name"], refresh=True)

        t0 = time.time()
        try:
            module = importlib.import_module(f"data_download.{ds['module']}")
            func = getattr(module, ds["function"])
            func()
            elapsed = time.time() - t0
            results[key] = {"status": "OK", "time": elapsed}
            logger.info(f"{ds['name']}: OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            results[key] = {"status": f"FAILED: {e}", "time": elapsed}
            logger.error(f"{ds['name']}: FAILED ({e})")

    total_elapsed = time.time() - t_total
    raw_dir = Path(__file__).resolve().parent / "raw"
    total_size = _dir_size_mb(raw_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Download Summary")
    print(f"{'='*60}")
    for key, res in results.items():
        name = DATASETS[key]["name"]
        status = res["status"]
        t = res["time"]
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {name:<40} {t:>6.1f}s  {status}")
    print(f"{'─'*60}")
    ok_count = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"  Total: {ok_count}/{len(results)} OK · {total_elapsed:.1f}s · {total_size:.1f} MB")
    print(f"{'='*60}\n")

    # Check for failures
    failures = [k for k, r in results.items() if r["status"] != "OK"]
    if failures:
        logger.warning(f"Some datasets failed: {failures}")
        logger.warning("Re-run the individual scripts to debug.")
        sys.exit(1)


def _check_dependencies():
    """Check that required packages are installed."""
    missing = []
    binary_incompatible = []

    for pkg, import_name in [("mne", "mne"), ("moabb", "moabb"), ("requests", "requests"),
                              ("tqdm", "tqdm"), ("numpy", "numpy")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
        except (AttributeError, ValueError) as e:
            # Binary incompatibility — package present but compiled for a different NumPy ABI
            # (common when NumPy 2.x is installed but packages were built for NumPy 1.x)
            binary_incompatible.append((pkg, str(e)))

    warned = False

    if binary_incompatible:
        warned = True
        print("\n⚠  Binary incompatibility detected (likely NumPy 2.x vs 1.x ABI mismatch):")
        for pkg, err in binary_incompatible:
            print(f"   • {pkg}: {err}")
        # Check installed numpy version
        try:
            import numpy as _np
            np_ver = _np.__version__
        except Exception:
            np_ver = "unknown"
        print(f"\n   Installed NumPy: {np_ver}")
        if np_ver.startswith("2"):
            print("   NumPy 2.x is installed, but some packages were compiled for NumPy 1.x.")
            print("   Fix options:")
            print('     1. Downgrade NumPy:          pip install "numpy<2"')
            print("     2. Upgrade affected packages: pip install --upgrade numexpr bottleneck h5py pandas moabb")
        print()

    if missing:
        warned = True
        print(f"\n⚠  Missing packages: {', '.join(missing)}")
        print(f"   Install with:  pip install -r requirements.txt\n")

    if warned:
        response = input("Continue anyway? [y/N] ")
        if response.lower() != "y":
            sys.exit(0)


if __name__ == "__main__":
    main()
