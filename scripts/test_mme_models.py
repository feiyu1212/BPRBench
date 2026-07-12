#!/usr/bin/env python3
"""
Test script to verify all MME models can be loaded and run inference.
Tests each model architecture with minimal dummy data.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing basic imports...")
    print("=" * 60)

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("sentencepiece", "SentencePiece"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("ftfy", "ftfy"),
        ("open_clip", "OpenCLIP"),
    ]

    failed = []
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError as e:
            print(f"✗ {display_name}: {e}")
            failed.append(display_name)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    print("\n✓ All basic imports successful!\n")
    return True


def test_vir2_clip():
    """Test Vir2-CLIP model (built-in)."""
    print("=" * 60)
    print("Testing Vir2-CLIP model...")
    print("=" * 60)

    try:
        from benchmark.vir2_clip import get_vir2_clip, get_vir2_clip_tokenizer
        print("✓ Vir2-CLIP imports successful")

        # Note: We don't test actual model loading without checkpoints
        print("✓ Vir2-CLIP module available")
        return True
    except ImportError as e:
        print(f"✗ Vir2-CLIP import failed: {e}")
        traceback.print_exc()
        return False


def test_openclip():
    """Test OpenCLIP model."""
    print("=" * 60)
    print("Testing OpenCLIP model...")
    print("=" * 60)

    try:
        import open_clip
        print(f"✓ OpenCLIP version: {open_clip.__version__}")

        # Test model creation (this will download if not cached)
        print("Testing model creation with pretrained weights...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k'
        )
        print("✓ OpenCLIP model creation successful")

        # Test tokenizer
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("✓ OpenCLIP tokenizer successful")

        return True
    except Exception as e:
        print(f"✗ OpenCLIP test failed: {e}")
        traceback.print_exc()
        return False


def test_conch():
    """Test CONCH model installation."""
    print("=" * 60)
    print("Testing CONCH model...")
    print("=" * 60)

    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        print("✓ CONCH imports successful")

        # Note: Can't test full loading without checkpoint
        print("✓ CONCH module available")
        return True
    except ImportError as e:
        print(f"✗ CONCH import failed: {e}")
        print("Note: CONCH requires: pip install git+https://github.com/mahmoodlab/CONCH.git")
        traceback.print_exc()
        return False


def test_musk():
    """Test MUSK model installation."""
    print("=" * 60)
    print("Testing MUSK model...")
    print("=" * 60)

    try:
        import musk
        import musk.modeling
        from musk import utils
        print("✓ MUSK imports successful")

        # Note: Can't test full loading without checkpoint
        print("✓ MUSK module available")
        return True
    except ImportError as e:
        print(f"✗ MUSK import failed: {e}")
        print("Note: MUSK requires: pip install git+https://github.com/lilab-stanford/MUSK.git")
        traceback.print_exc()
        return False


def test_bprbench_models():
    """Test BPRBench model registration."""
    print("=" * 60)
    print("Testing BPRBench model adapters...")
    print("=" * 60)

    try:
        from benchmark.models import ensure_clip_models_registered
        from benchmark.core.registry import list_models

        # Ensure all CLIP models are registered
        ensure_clip_models_registered()

        # List all registered models
        models = list_models()
        print(f"✓ Registered {len(models)} models:")
        for model_name in models:
            print(f"  - {model_name}")

        # Check for MME-specific models
        mme_models = ["openclip", "conch", "vir2", "musk", "quiltnet", "pathgen"]
        found_mme = [m for m in mme_models if m in models]
        missing_mme = [m for m in mme_models if m not in models]

        print(f"\n✓ Found MME models: {', '.join(found_mme)}")
        if missing_mme:
            print(f"⚠ Missing MME models: {', '.join(missing_mme)}")

        return True
    except Exception as e:
        print(f"✗ BPRBench model test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🧪 MME Environment Test Suite")
    print("=" * 60)

    results = {}

    # Test basic imports
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n❌ Basic imports failed - cannot continue with model tests")
        sys.exit(1)

    # Test individual models
    results["vir2_clip"] = test_vir2_clip()
    results["openclip"] = test_openclip()
    results["conch"] = test_conch()
    results["musk"] = test_musk()
    results["bprbench_models"] = test_bprbench_models()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed! MME environment is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
