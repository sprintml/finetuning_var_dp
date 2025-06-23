import os
import glob
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


def create_npz_from_sample_folder(
    sample_folder: str, path_to_save: str, type: str, image_size=(256, 256)
):
    """
    Builds a single .npz file from a folder of .png samples, ensuring all images have the same shape.
    """
    samples = []
    # Only search the current folder (non-recursive)
    pngs = glob.glob(os.path.join(sample_folder, "*.png"))
    assert len(pngs) > 0, f"No PNG files found in {sample_folder}"

    for png in tqdm(pngs, desc="Building .npz file from samples (png only)"):
        with Image.open(png) as sample_pil:
            # Ensure all images are RGB
            sample_pil = sample_pil.convert("RGB")
            # Resize to a fixed shape using LANCZOS (best quality)
            if sample_pil.size != image_size:
                sample_pil = sample_pil.resize(image_size, Image.LANCZOS)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)

    samples = np.stack(samples)  # Now all images should have the same shape

    # Ensure the directory exists
    os.makedirs(path_to_save, exist_ok=True)

    # Save the .npz file with correct path joining
    if type == "gen":
        npz_path = os.path.join(path_to_save, "generated_samples.npz")
    elif type == "ref":
        npz_path = os.path.join(path_to_save, "reference_samples.npz")
    else:
        raise ValueError(f"Invalid type: {type}")

    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
