import os, sys, enum, logging
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nb
import typer

from .noise_estimate import estimate_sigma
from ._tnlm import temporal_nl_means


log = logging.getLogger(__name__)

class SpatialWeight(enum.Enum):
    UNIFORM = enum.auto()
    GAUSSIAN = enum.auto()


def denoise_tnlm(
    in_data: np.ndarray, 
    sigma: float, 
    radius: int = 5, 
    mask: Optional[np.ndarray] = None,
    spatial_weight: SpatialWeight = SpatialWeight.UNIFORM,
):
    """Apply temporal nl-means denoising to a 4D array"""
    if mask is None:
        mask = np.ones(in_data.shape[:3], dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8, copy=False)
    return temporal_nl_means(in_data, sigma, mask, radius)


app = typer.Typer()


@app.command()
def denoise_tnlm_files(
    in_path: Path, 
    sigma: float = None, 
    radius: int = 5, 
    mask: Optional[str] = None, 
    out_path: Optional[Path] = None,
    verbose: bool = False,
    debug: bool = False,
):
    """Apply temporal nl-means denoising to a 4D image file"""
    # Setup logging
    LOG_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s"
    def_formatter = logging.Formatter(LOG_FORMAT)
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.DEBUG)
    stream_formatter = logging.Formatter("%(threadName)s %(name)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    if debug:
        stream_handler.setLevel(logging.DEBUG)
    elif verbose:
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.WARN)
    root_logger.addHandler(stream_handler)
    # Load data, process, save results
    in_img = nb.load(in_path)
    in_data = in_img.get_fdata()
    if mask is not None:
        mask = np.asarray(nb.load(mask).dataobj)
    if sigma is None:
        sigma = np.max(estimate_sigma(in_data))
        log.info("Using auto calculated sigma = %f", sigma)
    out_data = denoise_tnlm(in_data, sigma, radius, mask)
    out_img = nb.Nifti1Image(out_data, in_img.affine)
    if out_path is None:
        out_path = in_path.parent / f'denoised_{in_path.name}'
    nb.save(out_img, out_path)


if __name__ == "__main__":
    app()