from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from benchmark.core.logging import get_logger

log = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_IMAGES_PER_SAMPLE = 4 # 20  # 多图 sample 最多使用的图片数


class ImageIndexer:
    """
    Builds an in-memory index: folder_name -> List[Sample]
    Where a Sample is a List[Path].
    - If a file is found directly under folder_name -> Single-image sample [path]
    - If a subfolder is found -> Multi-image sample [path1, path2, ...]
    """

    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        # key: class folder name (e.g., "IDC")
        # value: List of samples (each sample is List[Path])
        self._cache: Dict[str, List[List[Path]]] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Image dir not found: {self.root}")

        log.info("Indexing samples in %s...", self.root)
        total_samples = 0
        total_images = 0

        # Iterate over class folders (e.g., data/images/IDC)
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue

            key = class_dir.name
            samples = []

            # Iterate over the contents of the class folder
            for item in sorted(class_dir.iterdir()):
                if item.name.startswith("."):
                    continue

                # Case A: Multi-Instance (Subdirectory)
                if item.is_dir():
                    # Recursively find images inside this specific sample folder
                    imgs = sorted([
                        p.resolve() for p in item.rglob("*")
                        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                    ])
                    if imgs:
                        imgs_limited = imgs[:MAX_IMAGES_PER_SAMPLE]
                        samples.append(imgs_limited)
                        total_images += len(imgs_limited)

                # Case B: Single-Instance (File)
                elif item.is_file() and item.suffix.lower() in _IMAGE_EXTS:
                    samples.append([item.resolve()])
                    total_images += 1

            if samples:
                self._cache[key] = samples
                total_samples += len(samples)

        log.info(
            "Indexed %d samples (total %d images) across %d folders",
            total_samples,
            total_images,
            len(self._cache),
        )

    def get_samples(self, folders: Iterable[str]) -> List[List[Path]]:
        """
        Retrieve all samples for the requested class folders.
        Returns: List[List[Path]]
        """
        all_samples: List[List[Path]] = []
        for folder in folders:
            raw_samples = self._cache.get(str(folder), [])

            # # --- DEBUG: Reduce data volume ---
            # # Strategy: Keep 1 single-image sample and 1 multi-image sample per folder.
            # singles = [s for s in raw_samples if len(s) == 1]
            # multis = [s for s in raw_samples if len(s) > 1]

            # selection = []
            # if singles:
            #     selection += singles[:3]
            # if multis:
            #     selection += multis[:3]
            
            # all_samples.extend(selection)
            # # --- END DEBUG ---

            all_samples.extend(raw_samples)

        # print(f"all_samples: {all_samples}")
        return all_samples