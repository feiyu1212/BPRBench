from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from benchmark.core.logging import get_logger
from benchmark.core.registry import register_model
from benchmark.models.base import BaseInference, PredictionRecord

log = get_logger(__name__)


def _to_torch_device(device: str) -> torch.device:
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    return d


def _collect_candidates_grouped(
    options: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[List[str]]]:
    """
    Group synonyms by their option label, preserving first-seen order of labels.

    Returns:
        unique_labels: e.g. ["A", "B", "C", "D"]
        grouped_texts:  parallel list of synonym lists, e.g. [["tumor", "neoplasm"], ["normal"], ...]

    This is the entry point for the paper's class-prototype aggregation strategy
    (Section 2.1, Fig. 2a): per-class synonyms are mean-pooled into a single
    unified prototype before cosine similarity with the image feature.
    """
    unique_labels: List[str] = []
    grouped_texts: List[List[str]] = []
    seen: Dict[str, int] = {}

    for opt in options:
        label = str(opt.get("label", "")).strip().upper()
        if not label:
            continue

        raw_candidates = opt.get("classnames") or [opt.get("text", "")]
        if isinstance(raw_candidates, str):
            raw_candidates = [raw_candidates]

        cleaned_texts = [str(c).strip() for c in raw_candidates if str(c).strip()]
        if not cleaned_texts:
            continue

        if label in seen:
            # Merge synonyms for a repeated label (rare but possible).
            grouped_texts[seen[label]].extend(cleaned_texts)
        else:
            seen[label] = len(unique_labels)
            unique_labels.append(label)
            grouped_texts.append(cleaned_texts)

    return unique_labels, grouped_texts


class _ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[Path], preprocess):
        self.paths = list(paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return str(path), self.preprocess(image)


class _ClipAdapterBase(BaseInference, ABC):
    def __init__(self, device: str, **kwargs: Any):
        super().__init__(device=device, **kwargs)
        self.torch_device = _to_torch_device(device)
        self.device = self.torch_device

    @abstractmethod
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(
        self,
        samples: List[List[Path]],
        question_text: str,
        options: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[PredictionRecord]:
        del question_text
        if not samples:
            return []

        import time as _time

        start_time = _time.time()
        batch_size = int(kwargs.get("batch_size", 64))
        num_workers = int(kwargs.get("num_workers", 4))
        correct_type = str(kwargs.get("correct_type", "max")).lower()

        # Paper-aligned aggregation: group synonyms per option label, then build
        # a single mean-pooled + L2-normalized prototype per class.
        unique_labels, grouped_texts = _collect_candidates_grouped(options)
        if not unique_labels:
            return []

        # --- Step 1: Flatten samples for batch processing ---
        flat_paths: List[Path] = []
        sample_map: List[int] = []  # Maps each flat image index to original sample index

        for sample_idx, img_list in enumerate(samples):
            for img_path in img_list:
                flat_paths.append(img_path)
                sample_map.append(sample_idx)

        if not flat_paths:
            return []

        # --- Step 2: Build per-class prototype embeddings ---
        # For each label, encode its synonyms -> mean pool -> L2 normalize -> prototype.
        with torch.no_grad():
            prototypes: List[torch.Tensor] = []
            for texts in grouped_texts:
                embs = self._encode_text(texts)  # (N_syn, Dim), already L2-normalized
                avg_emb = embs.mean(dim=0, keepdim=True)  # (1, Dim)
                avg_emb = F.normalize(avg_emb, dim=-1)  # re-normalize the prototype
                prototypes.append(avg_emb)
            text_emb = torch.cat(prototypes, dim=0)  # (N_labels, Dim)

        # --- Step 3: Encode all images ---
        ds = _ImagePathDataset(flat_paths, self._preprocess)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        all_img_embs_list = []
        with torch.no_grad():
            for _, images in dl:
                feats = self._encode_image(images.to(self.torch_device))  # (Batch, Dim)
                all_img_embs_list.append(feats)

        flat_img_tensor = torch.cat(all_img_embs_list, dim=0)  # (Total_Images, Dim)

        # --- Step 4: Average pooling (aggregation) ---
        num_samples = len(samples)
        dim = flat_img_tensor.shape[1]

        sample_embs = torch.zeros(
            (num_samples, dim), device=self.torch_device, dtype=flat_img_tensor.dtype
        )
        indices_tensor = torch.tensor(sample_map, device=self.torch_device, dtype=torch.long)
        sample_embs.index_add_(0, indices_tensor, flat_img_tensor)

        counts = torch.zeros((num_samples, 1), device=self.torch_device)
        ones = torch.ones((len(sample_map), 1), device=self.torch_device)
        counts.index_add_(0, indices_tensor, ones)
        counts[counts == 0] = 1
        sample_embs = sample_embs / counts
        sample_embs = F.normalize(sample_embs, dim=-1)

        # --- Step 5: Similarity against class prototypes ---
        sims = sample_embs @ text_emb.T
        if correct_type == "min":
            idxs = torch.argmin(sims, dim=1)
        else:
            idxs = torch.argmax(sims, dim=1)

        latency_per_sample = (_time.time() - start_time) / max(num_samples, 1)
        rows: List[PredictionRecord] = []
        for i, (idx, sim_row) in enumerate(zip(idxs.cpu().tolist(), sims.cpu().tolist())):
            sample_paths = samples[i]
            if len(sample_paths) > 1:
                identifier = str(sample_paths[0].parent)
            else:
                identifier = str(sample_paths[0])

            pred = unique_labels[idx]
            # Record prototype score so the run is auditable.
            raw = f"score={sim_row[idx]:.6f}|prototype_label={pred}"
            rows.append(
                PredictionRecord(
                    sample_id=identifier,
                    pred_label=pred,
                    raw_output=raw,
                    latency_s=latency_per_sample,
                )
            )

        return rows

    def predict_from_tensors(
        self,
        image_tensors: torch.Tensor,
        options: List[Dict[str, Any]],
        correct_type: str = "max",
    ) -> List[PredictionRecord]:
        """
        Predict from pre-loaded image tensors (skipping DataLoader/disk I/O).
        Used for efficiency benchmarking so inference time reflects pure GPU compute.
        image_tensors: (N_samples, C, H, W) preprocessed and on self.torch_device.
        """
        if image_tensors is None or image_tensors.numel() == 0:
            return []
        correct_type = str(correct_type).lower()
        if correct_type not in {"max", "min"}:
            correct_type = "max"

        import time as _time

        start_time = _time.time()
        unique_labels, grouped_texts = _collect_candidates_grouped(options)
        if not unique_labels:
            return []

        with torch.no_grad():
            prototypes: List[torch.Tensor] = []
            for texts in grouped_texts:
                embs = self._encode_text(texts)
                avg_emb = embs.mean(dim=0, keepdim=True)
                avg_emb = F.normalize(avg_emb, dim=-1)
                prototypes.append(avg_emb)
            text_emb = torch.cat(prototypes, dim=0)

            img_emb = self._encode_image(image_tensors)
            sample_embs = F.normalize(img_emb, dim=-1)
            sims = sample_embs @ text_emb.T

        if correct_type == "min":
            idxs = torch.argmin(sims, dim=1)
        else:
            idxs = torch.argmax(sims, dim=1)

        num_samples = image_tensors.shape[0]
        latency_per_sample = (_time.time() - start_time) / max(num_samples, 1)
        rows: List[PredictionRecord] = []
        for i in range(num_samples):
            idx_i = idxs[i].item()
            pred = unique_labels[idx_i]
            raw = f"score={sims[i, idx_i].item():.6f}|prototype_label={pred}"
            rows.append(
                PredictionRecord(
                    sample_id=f"sample_{i}",
                    pred_label=pred,
                    raw_output=raw,
                    latency_s=latency_per_sample,
                )
            )
        return rows


@register_model("clip_openclip")
@register_model("openclip")
@register_model("quiltnet")  # alias: QuiltNet is OpenCLIP with ViT-B-32 + fine-tuned weights
@register_model("pathgen")   # alias: PathGen is OpenCLIP with ViT-B-16 + fine-tuned weights
class OpenCLIPAdapter(_ClipAdapterBase):
    """
    General purpose adapter using the open_clip library.
    Can load trained OpenCLIP models, original OpenAI weights, or fine-tunes like QuiltNet/PathGen.
    """

    def __init__(self, device: str, arch: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", checkpoint: str | None = None, **kwargs: Any):
        super().__init__(device=device, **kwargs)

        import open_clip

        self.arch = arch
        self._open_clip = open_clip
        weight = checkpoint or pretrained
        log.info("Loading OpenCLIP arch=%s weight=%s", arch, weight)
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=weight)
        self.model = model.to(self.torch_device).eval()
        self.preprocess_fn = preprocess
        self.tokenizer = open_clip.get_tokenizer(arch)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess_fn(image)

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.torch_device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)


@register_model("clip_vir2")
@register_model("vir2")
@register_model("vir2_clip")
class Vir2Adapter(_ClipAdapterBase):
    def __init__(self, device: str, checkpoint: str, remove_text: bool = False, remove_visual: bool = False, **kwargs: Any):
        super().__init__(device=device, **kwargs)

        try:
            from benchmark.vir2_clip import get_vir2_clip, get_vir2_clip_tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Vir2_CLIP is not available. It ships inside this repo as the "
                "`benchmark.vir2_clip` subpackage; run `pip install -e \".[clip]\"` "
                "to install it."
            ) from exc

        self.tokenizer = get_vir2_clip_tokenizer()
        model, preprocess = get_vir2_clip(
            checkpoint=checkpoint,
            remove_text=remove_text,
            remove_visual=remove_visual,
        )
        self.model = model.to(self.torch_device).eval()
        self.preprocess_fn = preprocess

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess_fn(image)

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.torch_device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)


@register_model("clip_conch")
@register_model("conch")
class ConchAdapter(_ClipAdapterBase):
    def __init__(self, device: str, checkpoint: str, model_cfg: str = "conch_ViT-B-16", force_image_size: int = 224, **kwargs: Any):
        super().__init__(device=device, **kwargs)

        try:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
        except ImportError as exc:
            raise RuntimeError(
                "CONCH is not installed. Please install via: pip install git+https://github.com/mahmoodlab/CONCH.git"
            ) from exc

        self._tokenize = tokenize
        self._raw_tokenizer = get_tokenizer()
        model, preprocess = create_model_from_pretrained(
            model_cfg,
            checkpoint_path=checkpoint,
            force_image_size=force_image_size,
        )
        self.model = model.to(self.torch_device).eval()
        self.preprocess_fn = preprocess

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess_fn(image)

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self._tokenize(texts=texts, tokenizer=self._raw_tokenizer).to(self.torch_device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)


@register_model("clip_musk")
@register_model("musk")
class MuskAdapter(_ClipAdapterBase):
    def __init__(self, device: str, checkpoint: str, tokenizer: str, model_name: str = "musk_large_patch16_384", max_text_len: int = 100, **kwargs: Any):
        super().__init__(device=device, **kwargs)

        try:
            from musk import utils
            import musk.modeling  # register musk_large_patch16_384 with timm
            from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
            from timm.models import create_model
            from transformers import XLMRobertaTokenizer
            import torchvision.transforms as transforms
        except ImportError as exc:
            raise RuntimeError(
                "MUSK is not installed. Please install via: pip install git+https://github.com/lilab-stanford/MUSK.git"
            ) from exc

        self._utils = utils
        self.max_text_len = int(max_text_len)

        log.info("Loading MUSK checkpoint=%s tokenizer=%s", checkpoint, tokenizer)
        model = create_model(model_name)
        utils.load_model_and_may_interpolate(checkpoint, model, "model|module", "")
        self.model = model.to(self.torch_device).eval()
        self.tokenizer = XLMRobertaTokenizer(tokenizer)

        self.preprocess_fn = transforms.Compose(
            [
                transforms.Resize(384, interpolation=3, antialias=True),
                transforms.CenterCrop((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess_fn(image)

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        text_ids = []
        paddings = []
        for txt in texts:
            ids, pads = self._utils.xlm_tokenizer(txt, self.tokenizer, max_len=self.max_text_len)
            text_ids.append(torch.tensor(ids).unsqueeze(0))
            paddings.append(torch.tensor(pads).unsqueeze(0))

        text_tensor = torch.cat(text_ids).to(self.torch_device)
        pad_tensor = torch.cat(paddings).to(self.torch_device)

        text_emb = self.model(
            text_description=text_tensor,
            padding_mask=pad_tensor,
            with_head=True,
            out_norm=True,
        )[1]
        return text_emb

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        img_emb = self.model(
            image=images,
            with_head=True,
            out_norm=True,
        )[0]
        return img_emb
