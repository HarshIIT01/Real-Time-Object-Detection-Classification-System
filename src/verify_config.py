"""
verify_config.py — Diagnostic tool for Issue I.

Detects mismatches between configs/config.yaml (the intended settings) and
the args.yaml produced by the last YOLO training run.

Usage:
    python -m src.verify_config
    python -m src.verify_config --config configs/config.yaml --run outputs/models/yolov8_run
"""
import os
import sys
import argparse
import json
import yaml

# Fields to compare, as (args.yaml key, config path, config default)
# config path is a dot-separated key into the loaded config dict.
COMPARISON_MAP = [
    # (args_yaml_key,     config_key_path,                   config_default)
    ("imgsz",             "data.detection_imgsz",             640),
    ("cos_lr",            "training.detection.cos_lr",        True),
    ("mixup",             "training.detection.mixup",         0.15),
    ("copy_paste",        "training.detection.copy_paste",    0.1),
    ("mosaic",            "training.detection.mosaic",        1.0),
    ("epochs",            "training.epochs",                  100),
    ("patience",          "training.detection.patience",      30),
    ("lr0",               "training.detection.lr0",           0.01),
    ("lrf",               "training.detection.lrf",           0.01),
    ("momentum",          "training.detection.momentum",      0.937),
    ("weight_decay",      "training.detection.weight_decay",  0.0005),
    ("warmup_epochs",     "training.detection.warmup_epochs", 3.0),
    ("hsv_h",             "training.detection.hsv_h",         0.015),
    ("hsv_s",             "training.detection.hsv_s",         0.7),
    ("hsv_v",             "training.detection.hsv_v",         0.4),
    ("fliplr",            "training.detection.fliplr",        0.5),
    ("erasing",           "training.detection.erasing",       0.4),
    ("close_mosaic",      "training.detection.close_mosaic",  10),
]


def _get_nested(d, dotted_key, default=None):
    """Retrieve a value from a nested dict using a dotted key string."""
    keys = dotted_key.split(".")
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def verify_config(config_path="configs/config.yaml", run_dir="outputs/models/yolov8_run"):
    """
    Compare config.yaml vs the last yolov8_run/args.yaml.

    Prints a side-by-side table and returns a list of mismatch dicts.
    Exits with code 1 if critical mismatches (imgsz, cos_lr, mixup) are found.
    """
    # ── Load config.yaml ──────────────────────────────────────────────────────
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # -- Load args.yaml --------------------------------------------------------
    args_yaml_path = os.path.join(run_dir, "args.yaml")
    if not os.path.exists(args_yaml_path):
        print(f"[WARN] No args.yaml found at '{args_yaml_path}'.")
        print(f"       -> No previous training run to compare against.")
        print(f"       -> Run 'python -m src.train' first.")
        return []

    with open(args_yaml_path) as f:
        args = yaml.safe_load(f)

    # -- Compare ---------------------------------------------------------------
    CRITICAL_KEYS = {"imgsz", "cos_lr", "mixup", "copy_paste"}

    mismatches = []
    rows = []

    col_w = [28, 22, 22, 8]

    header = (
        f"{'Parameter':<{col_w[0]}} {'config.yaml':<{col_w[1]}} {'args.yaml (actual)':<{col_w[2]}} {'Status'}"
    )
    separator = "-" * sum(col_w)

    print()
    print("=" * sum(col_w))
    print("  CONFIG vs TRAINING RUN RECONCILIATION")
    print(f"  Config : {os.path.abspath(config_path)}")
    print(f"  Run    : {os.path.abspath(args_yaml_path)}")
    print("=" * sum(col_w))
    print(header)
    print(separator)

    for args_key, config_key, default in COMPARISON_MAP:
        cfg_val  = _get_nested(config, config_key, default)
        args_val = args.get(args_key, "MISSING")

        # Normalize for comparison (handle float vs int, True/False vs true/false)
        def _norm(v):
            if isinstance(v, bool):
                return v
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        match = _norm(cfg_val) == _norm(args_val)

        if not match:
            critical = args_key in CRITICAL_KEYS
            status = "[CRITICAL]" if critical else "[MISMATCH]"
            mismatches.append({
                "key": args_key,
                "config_val": cfg_val,
                "args_val": args_val,
                "critical": critical,
            })
        else:
            status = "[OK]"

        row = (
            f"{args_key:<{col_w[0]}} "
            f"{str(cfg_val):<{col_w[1]}} "
            f"{str(args_val):<{col_w[2]}} "
            f"{status}"
        )
        print(row)

    print(separator)

    # -- Summary ---------------------------------------------------------------
    critical_mismatches = [m for m in mismatches if m["critical"]]
    warn_mismatches     = [m for m in mismatches if not m["critical"]]

    if not mismatches:
        print("\n[OK] All checked parameters match. Config is consistent with last run.")
    else:
        print(f"\n  Found {len(mismatches)} mismatch(es): "
              f"{len(critical_mismatches)} critical, {len(warn_mismatches)} warnings.")

    if critical_mismatches:
        print()
        print("[CRITICAL MISMATCHES] (these directly degrade model quality):")
        for m in critical_mismatches:
            print(f"  {m['key']:20} config wants {m['config_val']!r:12} but run used {m['args_val']!r}")
        print()
        print("FIX: Always use  python -m src.train --config configs/config.yaml")
        print("     Do NOT pass YOLO args manually on the command line.")

    print()

    # -- Also check model size -------------------------------------------------
    cfg_model = _get_nested(config, "model.yolo_model", "yolov8s.pt")
    run_model  = args.get("model", "MISSING")
    if cfg_model != run_model:
        print(f"[WARN] Model size mismatch: config wants '{cfg_model}' but run used '{run_model}'")
        print(f"       Larger models (s/m/l/x) give better accuracy than nano (n).")
        print()

    # -- Dataset ---------------------------------------------------------------
    cfg_data = _get_nested(config, "data.detection_dataset", "coco128.yaml")
    run_data  = args.get("data", "MISSING")
    if cfg_data != run_data:
        print(f"[WARN] Dataset mismatch: config='{cfg_data}' run='{run_data}'")
        print()

    return mismatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify config.yaml matches last training run")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run",    default="outputs/models/yolov8_run",
                        help="Path to yolov8_run directory containing args.yaml")
    args = parser.parse_args()

    mismatches = verify_config(config_path=args.config, run_dir=args.run)
    critical = [m for m in mismatches if m["critical"]]
    sys.exit(1 if critical else 0)
