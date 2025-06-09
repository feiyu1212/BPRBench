"""
- read slide using OpenSlideAPI
- split to N * N patches
- filter background patches, and keep the tissue patches
"""
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import glob
import shutil



import numpy as np

from utils.openslide_api import OpenSlideAPI

_DEFAULT_PATCH_DIR = 'chat_data/patches'


def is_background(patch, threshold=220, background_ratio=0.3):
    """
    Determine if the patch is a background patch.
    :param patch: PIL image of the patch
    :param threshold: Pixel intensity threshold to distinguish between background and tissue
    :param background_ratio: The ratio of background pixels to total pixels above which a patch is considered background
    :return: True if the patch is a background patch, False otherwise
    """
    gray = patch.convert('L')  # Convert to grayscale
    bw = np.array(gray) > threshold
    if np.mean(bw) > background_ratio:
        return True
    return False

# slide_path, region, level = OpenSlideAPI.parse_region("http://sdfsdf/region/openslide/{self.filename}/3/2/5/8/1")


def load_slide_patches(slide_path, level=0, patch_size=224, num_threads=8, num_patches=None, save_dir=None, check_exists=False, region=None):
    """
    Read a slide and split it into non-background patches using multi-threading.
    :param slide_path: Path to the whole-slide image
    :param level: Level of the slide to read
    :param patch_size: Size of the patches to split into (224x224 by default)
    :param save_dir: Directory to save patches
    :param num_threads: Number of threads to use
    :param num_patches: Maximum number of patches to save (None for no limit)
    :return: patch_dir
    """
    if not save_dir:
        save_dir = _DEFAULT_PATCH_DIR

    # Create patch directory
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    if region is not None:
        patch_dir = os.path.join(save_dir, 'region', slide_name, 'NORM')
        if os.path.exists(patch_dir):
            shutil.rmtree(patch_dir)
    else:
        # patch_dir = os.path.join(save_dir, slide_name, 'NORM')
        patch_dir = os.path.join(save_dir, slide_name)
    if check_exists and num_patches:
        num_exists = len(glob.glob(patch_dir + '/*.jpg'))
        if num_exists >= num_patches:
            print(f"exists {num_exists} patches at {patch_dir}")
            return patch_dir.rstrip('/NORM')
        else:
            if os.path.exists(patch_dir):
                shutil.rmtree(patch_dir)
    else:
        if os.path.exists(patch_dir):
            shutil.rmtree(patch_dir)
    os.makedirs(patch_dir, exist_ok=True)

    # Initialize slide instance once
    slide = OpenSlideAPI(slide_path)
    # print(slide.level_dimensions)

    downsample = slide.level_downsamples[level]
    print(f"Slide dimensions: {slide.dimensions} downsample: {downsample}")

    if region is not None:
        rx, ry, rw, rh, rlevel = region
        rds = slide.level_downsamples[rlevel]
        f = rds / downsample
        x_start, y_start, width, height = int(rx * f), int(ry * f), max(int(rw * f), patch_size), max(int(rh * f), patch_size)
    else:
        x_start, y_start = 0, 0
        width, height = slide.level_dimensions[level]    

    # Prepare list of patch coordinates
    patch_coords = [
        (x, y)
        for y in range(y_start, height, patch_size)
        for x in range(x_start, width, patch_size)
        if x + patch_size <= width and y + patch_size <= height
    ]

    print(f"Total patches to process: {len(patch_coords)}")

    # Lock for thread-safe slide access and saving
    # slide_lock = threading.Lock()
    counter_lock = threading.Lock()
    saved_patches = 0  # Shared counter for saved patches

    def process_patch(coord):
        nonlocal saved_patches
        if saved_patches >= num_patches:
            return False
        x, y = coord
        try:
            # with slide_lock:
            patch = slide.read_region((x * downsample, y * downsample), level, (patch_size, patch_size)).convert('RGB')
            if not is_background(patch):
                with counter_lock:
                    if num_patches is not None and saved_patches >= num_patches:
                        return False  # Reached the limit, do not save more patches
                    saved_patches += 1
                patch_filename = f"{level}_{x}_{y}.jpg"
                patch.save(os.path.join(patch_dir, patch_filename))
                return True
            return False
        except Exception as e:
            print(f"Error processing patch ({x}, {y}): {e}")
            return False

    # Use ThreadPoolExecutor to process patches in parallel
    processed = 0
    saved = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_coord = {executor.submit(process_patch, coord): coord for coord in patch_coords}
        for future in as_completed(future_to_coord):
            processed += 1
            try:
                result = future.result()
                if result:
                    saved += 1
                    if num_patches is not None and saved >= num_patches:
                        print(f"Reached the maximum number of patches: {saved}")
                        break  # Exit the loop once the limit is reached
            except Exception as e:
                coord = future_to_coord[future]
                print(f"Exception for patch {coord}: {e}")
            if processed % 1000 == 0:
                print(f"Processed {processed}/{len(patch_coords)} patches, Saved: {saved}")
    
    print(f"Finished processing. Total patches saved: {saved}")
    return patch_dir.rstrip('/NORM')



def load_slide_patches_all(slide_path, level=0, patch_size=224, num_threads=16, save_dir=None):
    """
    Read a slide at a given level and split it into non-background patches. 
    All non-background patches are generated from the full slide (no region support).

    :param slide_path:  Path to the whole-slide image file
    :param level:       Level of the slide to read (default: 0)
    :param patch_size:  Size of each patch (default: 224)
    :param num_threads: Number of threads for parallel patch extraction (default: 8)
    :param save_dir:    Directory to save the extracted patches (optional)
    :return:            The directory where patches are saved
    """
    if save_dir is None:
        save_dir = "./patches"

    # Create patch directory based on the slide name
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    patch_dir = os.path.join(save_dir, slide_name)
    if os.path.exists(patch_dir):
        shutil.rmtree(patch_dir)
    os.makedirs(patch_dir, exist_ok=True)

    # Initialize the slide
    slide = OpenSlideAPI(slide_path)
    downsample = slide.level_downsamples[level]

    # Determine the slide dimensions at the requested level
    width, height = slide.level_dimensions[level]

    # Generate all valid top-left coordinates for patch extraction
    patch_coords = [
        (x, y)
        for y in range(0, height, patch_size)
        for x in range(0, width, patch_size)
        if x + patch_size <= width and y + patch_size <= height
    ]

    def process_patch(coord):
        """Extract and save the patch if it is not background. Return 1 if saved, 0 otherwise."""
        x, y = coord
        try:
            # Read patch at the specified level and convert to RGB
            patch = slide.read_region(
                (int(x * downsample), int(y * downsample)),  # (X, Y) at level 0
                level,
                (patch_size, patch_size)
            ).convert('RGB')

            # Save patch only if it's not background
            if not is_background(patch):
                patch_filename = f"{level}_{x}_{y}.jpg"
                patch_path = os.path.join(patch_dir, patch_filename)
                patch.save(patch_path)
                return 1
        except Exception as e:
            # You can add logging here if you want to track errors
            pass
        return 0

    # Use a thread pool to process all patches in parallel
    saved_count = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_patch, coord) for coord in patch_coords]
        for future in as_completed(futures):
            saved_count += future.result()

    # Optionally, print summary
    print(f"Total non-background patches saved: {saved_count}")

    return patch_dir


