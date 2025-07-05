# BPRBench

## 📄 Overview

**BPRBench** is the first benchmark to evaluate MMs for breast pathology report generation. BPRBench encompasses 12 critical tasks involved in breast pathology reporting and 17,597 pathologist‑annotated image–option pairs derived from 377 breast‑cancer whole‑slide images, providing a unified testbed on which we benchmarked 11 pathology-specialized multimodal large language models (MLLMs, e.g., GPT-4o) and five general-purpose multimodal foundation models (MFMs, e.g., CONCH).

## 🚀 Quick Start

### Prerequisites

#### System Requirements

##### Hardware
*   **RAM**: Minimum 256 GB RAM recommended
*   **GPU**: NVIDIA A800 GPU

##### Software & Dependencies
The code was developed and tested on the following environment:
*   **Operating System**: Ubuntu 22.04 LTS
*   **CUDA Version**: 12.2


```bash
# Install required packages
pip install -r requirements.txt
```

### Running Evaluations

1. **Single Model Evaluation**:
```bash
cd BPRBench
python eval_tasks.py [model_index]
```

Where `model_index` corresponds to the model in the models list (0-based indexing).

2. **Calculate Metrics**:
```bash
python eval_metrics.py
python eval_sens_spec.py
```

## 📊 Benchmark Tasks

The benchmark includes **26 different tasks** covering:

### Core Tasks
1. **Cancer Detection** (Binary classification)
   - Carcinoma vs. Benign tissue
   
2. **Cancer Subtype Classification**
   - IDC (Invasive Ductal Carcinoma)
   - DCIS (Ductal Carcinoma In Situ)
   - ILC (Invasive Lobular Carcinoma)
   - Various rare subtypes

3. **Pathological Features**
   - Mitosis detection
   - Necrosis identification
   - Immune infiltration assessment
   - Architectural patterns

### Question Format
Each task includes:
- **Question text**: Natural language description
- **Multiple choice options**: 2-7 options per question
- **Class mappings**: Folder names to semantic classes
- **Prompt templates**: For CLIP-based evaluation

Example:
```json
{
    "id": 1,
    "question": "Does this image contain cancer?",
    "options": [
        {
            "label": "A",
            "text": "Yes",
            "classnames": ["Carcinoma"],
            "folders": ["IDC", "DCIS", "ILC", ...]
        },
        {
            "label": "B", 
            "text": "No",
            "classnames": ["Benign"],
            "folders": ["non-tumor"]
        }
    ]
}
```

## 📈 Evaluation Metrics

### Output Files
- `total_accuracy_per_model.csv`: Overall model rankings
- `average_accuracy_per_model_question_id.csv`: Per-task results
- `average_senspec_per_model_tag.csv`: Sensitivity/specificity analysis
- `unique_df.csv`: Detailed per-image predictions

## 🔧 Adding New Models

### CLIP-based Models
1. Create a new model class inheriting from `BaseClip`:
```python
from model.base_clip import BaseClip

class YourCLIPModel(BaseClip):
    def __init__(self, ckpt):
        super().__init__(ckpt)
        self.model_name = 'your-model'
    
    def _init_model(self, ckpt):
        # Load your model here
        pass
```

2. Add to the models list in `eval_tasks.py`

### Vision-Language Models
1. Create a model class with a `generate` method:
```python
class YourVLM:
    def __init__(self):
        # Initialize your model
        pass
    
    def generate(self, system, prompt, image_paths):
        # Return model's text response
        return response
```

2. Add to the models list in `eval_tasks.py`

## 💾 Data Format

### Results Format
Each model generates a CSV with:
- `file`: Image file path
- `pred_option`: Predicted option (A, B, C, D)
- `question_id`: Task identifier
- Additional model-specific columns
