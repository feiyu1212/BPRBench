# MME Environment Comparison

## Overview
This document compares the full `bprbench` environment with the minimal `mme_env` environment designed specifically for running MME (Multimodal Embodied) models.

## Full bprbench Environment

The full environment contains **~180 packages** including:
- Complete PyTorch stack with CUDA support
- Server dependencies (FastAPI, Uvicorn, etc.)
- Development tools
- Data science libraries (scikit-learn, scipy, h5py, etc.)
- Many utility libraries not essential for MME inference

### Major Package Categories in bprbench (187 total)

**Core ML/Deep Learning:**
- torch==2.2.0
- torchvision==0.17.0
- numpy==1.26.1
- pillow==10.4.0

**MME-Specific:**
- open-clip-torch==2.30.0
- transformers==4.48.3
- sentencepiece==0.2.1
- timm==1.0.24
- einops==0.8.2
- ftfy==6.3.1
- regex==2026.1.15
- conch (from GitHub)
- musk==1.0.0 (from GitHub)

**Server/Web (Not needed for MME):**
- fastapi==0.128.4
- uvicorn==0.40.0
- httpx==0.28.1
- requests==2.32.5
- starlette==0.52.1

**Data Science (Not needed for MME inference):**
- scikit-learn==1.7.2
- scipy==1.15.2
- h5py==3.15.1
- hdf5==1.14.6

**System/Development (Not needed):**
- git
- perl
- Various low-level libraries

## Minimal mme_env Environment

The minimal environment contains **~20 essential packages** focused solely on MME model inference:

### Essential Dependencies

**Core ML Framework:**
- python=3.10
- pytorch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0
- pillow>=10.0.0

**MME Model Requirements:**
- open-clip-torch>=2.23.0 (OpenCLIP, QuiltNet, PathGen)
- transformers>=4.30.0,<5 (Text processing)
- sentencepiece (Tokenization)
- timm>=0.9.0 (Vision models)
- einops (Tensor operations)
- ftfy (Text encoding)
- regex (Pattern matching)

**BPRBench Framework:**
- pandas>=2.0.0
- pyyaml>=6.0
- pydantic>=2.0
- python-dotenv>=1.0.0
- tqdm>=4.65.0

**External MME Models:**
- git+https://github.com/mahmoodlab/CONCH.git
- git+https://github.com/lilab-stanford/MUSK.git

## What Was Removed

The following categories were removed as they are not needed for MME model inference:

1. **Server infrastructure** - FastAPI, Uvicorn, HTTP clients
2. **Data science libraries** - scikit-learn, scipy (not used in inference)
3. **Scientific data formats** - h5py, hdf5
4. **Development tools** - git, perl
5. **Redundant utilities** - Many small libraries pulled as dependencies

## Size Comparison

- **Full bprbench**: ~180 packages
- **Minimal mme_env**: ~20 packages  
- **Reduction**: ~88% fewer packages

## Verified MME Models

All MME models have been verified to work with the minimal environment:
- ✅ OpenCLIP (openclip, quiltnet, pathgen)
- ✅ Vir2-CLIP (vir2, vir2_clip, clip_vir2)
- ✅ CONCH (conch, clip_conch)
- ✅ MUSK (musk, clip_musk)

## Usage

### Create the minimal environment:
```bash
conda env create -f mme_environment.yaml
conda activate mme_env
```

### Install BPRBench:
```bash
pip install -e .
```

### Test the environment:
```bash
python scripts/test_mme_models.py
```

## Notes

1. **CUDA Support**: The environment includes PyTorch with CUDA support if available
2. **Model Checkpoints**: Separate from environment - need to be downloaded separately
3. **Compatibility**: Python 3.10+ required
4. **Updates**: Package versions use minimum version constraints where appropriate