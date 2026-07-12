#!/usr/bin/env python3
"""Download / stage model checkpoints for BPRBench's local models.

This is a convenience wrapper around `huggingface_hub.snapshot_download` that
places every weight under a single `checkpoints/` directory at the repo root,
so the run wrappers (`scripts/run_mme_local.sh`, `scripts/run_mllm_qllava.sh`)
can locate them without any per-user env vars.

Models handled:
  * QuiltNet        wisdomik/QuiltNet-B-32          (public)
  * PathGen-CLIP    jamessyx/PathGen-CLIP           (gated)
  * CONCH           MahmoodLab/CONCH                (gated)
  * MUSK            xiangjx/musk                    (gated)
  * Quilt-LLaVA     wisdomik/Quilt-Llava-v1.5-7b    (public, optional pre-fetch)
  * Vir2-CLIP       Google Drive                     (public, opt-in — see README §2)

Gated models (CONCH, MUSK, PathGen-CLIP) require:
  1. A Hugging Face account that has accepted the model's access terms on the
     model page (links printed below).
  2. A login token. Either `export HF_TOKEN=...` or run `huggingface-cli login`
     once before this script.

Usage:
    python tools/download_checkpoints.py            # MME weights (Vir2-CLIP excluded)
    python tools/download_checkpoints.py vir2       # also fetch Vir2-CLIP from Google Drive
    python tools/download_checkpoints.py conch musk # only specific models
    python tools/download_checkpoints.py --skip-qllava   # skip the large LLaVA
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO_ROOT / "checkpoints"


@dataclass
class ModelSpec:
    key: str
    repo_id: str
    target: Path
    gated: bool
    url: str
    optional: bool = False


SPECS = [
    ModelSpec("quiltnet", "wisdomik/QuiltNet-B-32",
              CKPT_ROOT / "quiltnet", gated=False,
              url="https://huggingface.co/wisdomik/QuiltNet-B-32"),
    ModelSpec("pathgen", "jamessyx/PathGen-CLIP",
              CKPT_ROOT / "pathgen", gated=True,
              url="https://huggingface.co/jamessyx/PathGen-CLIP"),
    ModelSpec("conch", "MahmoodLab/CONCH",
              CKPT_ROOT / "conch", gated=True,
              url="https://huggingface.co/MahmoodLab/CONCH"),
    ModelSpec("musk", "xiangjx/musk",
              CKPT_ROOT / "musk", gated=True,
              url="https://huggingface.co/xiangjx/musk"),
    ModelSpec("qllava", "wisdomik/Quilt-Llava-v1.5-7b",
              CKPT_ROOT / "qllava", gated=False,
              url="https://huggingface.co/wisdomik/Quilt-Llava-v1.5-7b",
              optional=True),
]

VIR2_PATH = CKPT_ROOT / "vir2_clip" / "vir2_clip.pt"
VIR2_GDRIVE_ID = "1K2ZUSapvzKyyFkHROkzQ4yepwFvRIgxv"


def _import_hf_hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
        return True
    except ImportError:
        print(
            "ERROR: `huggingface_hub` is not installed.\n"
            "  Install it with:  pip install huggingface_hub\n"
            "  (it is already available inside the bpr_main / bpr_custom envs).",
            file=sys.stderr,
        )
        return False


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_one(spec: ModelSpec) -> str:
    """Download a single HF repo. Returns a status string."""
    from huggingface_hub import snapshot_download
    spec.target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(spec.target),
            token=_token(),
        )
        return "OK"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if any(s in msg for s in ("401", "403", "gated", "access", "forbidden", "unauthorized")):
            return (
                f"GATED/DENIED — request access at {spec.url}, then "
                f"`huggingface-cli login` (or export HF_TOKEN) and re-run."
            )
        return f"FAILED: {exc}"


def download_vir2() -> str:
    """Download Vir2-CLIP from Google Drive via gdown."""
    if VIR2_PATH.exists():
        return "OK"
    try:
        import gdown  # noqa: F401
    except ImportError:
        return (
            "FAILED: `gdown` is not installed. Install it with "
            "`pip install gdown` (available in the bpr_main / bpr_custom envs), "
            "then re-run."
        )
    VIR2_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(
            id=VIR2_GDRIVE_ID,
            output=str(VIR2_PATH),
            quiet=False,
        )
    except Exception as exc:  # noqa: BLE001
        return f"FAILED: {exc}"
    return "OK" if VIR2_PATH.exists() else "FAILED: download produced no file"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*",
                        help="Optional subset of model keys to download "
                             "(quiltnet, pathgen, conch, musk, qllava, vir2). "
                             "Default: all except vir2 (Vir2-CLIP has its own "
                             "manual-download quick start in the README).")
    parser.add_argument("--skip-qllava", action="store_true",
                        help="Skip the large Quilt-LLaVA repo (it auto-downloads "
                             "on first server start anyway).")
    args = parser.parse_args()

    if not _import_hf_hub():
        return 1

    requested = {m.lower() for m in args.models} if args.models else None
    results: list[tuple[str, str]] = []

    for spec in SPECS:
        if requested and spec.key not in requested:
            continue
        if spec.key == "qllava" and args.skip_qllava and not requested:
            results.append((spec.key, "skipped (--skip-qllava)"))
            continue

        print(f"\n[{spec.key}] {spec.repo_id} -> {spec.target.relative_to(REPO_ROOT)}")
        if spec.gated and not _token():
            print("  (gated model — no HF_TOKEN detected; will attempt and report "
                  "guidance on failure)")
        status = download_one(spec)
        print(f"  -> {status}")
        results.append((spec.key, status))

    # Vir2-CLIP (Google Drive) — opt-in only. It has its own manual-download quick
    # start in the README (§2 Step 2), so it is excluded from the default bulk
    # download. Fetch it explicitly with: python tools/download_checkpoints.py vir2
    if requested and "vir2" in requested:
        print(f"\n[vir2] Google Drive -> {VIR2_PATH.relative_to(REPO_ROOT)}")
        status = download_vir2()
        print(f"  -> {status}")
        results.append(("vir2", status))
    elif not requested:
        print("\n[vir2] skipped by default — see README §2 Step 2 for the manual "
              "download, or re-run with `python tools/download_checkpoints.py vir2`.")

    # Summary table.
    print("\n" + "=" * 64)
    print("Checkpoint status summary")
    print("=" * 64)
    for key, status in results:
        flag = "OK" if status.startswith("OK") else "!!"
        print(f"  [{flag}] {key:<10} {status}")
    print("=" * 64)
    print("Weights live under checkpoints/. They are gitignored; do not commit them.")

    failed = [k for k, s in results if not s.startswith("OK") and not s.startswith("skipped")]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
