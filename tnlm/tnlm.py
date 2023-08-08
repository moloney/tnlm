import os, sys, enum, logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nb
import typer
from scipy.ndimage import median_filter, gaussian_filter
from rich.console import Console

from ._tnlm import temporal_nl_means, estimate_neigh_bias_and_noise


log = logging.getLogger(__name__)


def robust_mean(arr, iqr_coeff=1.5):
    '''Take a mean value that ignores outliers

    Values are thresholed by the distance to the mean. The threshold is
    given by multiplying inter-quartile range by the `iqr_coeff`.
    '''
    first_quart, third_quart = np.percentile(arr, (25, 75))
    thresh = (third_quart - first_quart) * iqr_coeff
    if thresh == 0.0:
        mask = np.ones(arr.shape, dtype=np.bool8)
    else:
        mask = (arr > first_quart - thresh) & (arr < third_quart + thresh)
    return np.mean(arr[mask])


def estimate_bias_and_noise(
    in_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    radius: int = 5,
    n_best: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate local bias / noise from similar neigboring voxels in `in_data`
    
    The signal from each voxel (within `mask`) is compared to all neighbors within the
    `radius` (and `mask`) and a "bias" (mean of residuals) and "noise" (scaled stddev 
    of the residuals) is computed. The `n_best` (lowest) values are returned for each
    voxel.
    """
    if mask is None:
        mask = np.ones(in_data.shape[:3], dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8, copy=False)
    best_bias, best_noise = estimate_neigh_bias_and_noise(
        in_data.astype(np.float32), mask, radius, n_best
    )
    # TODO: Handle -1 results here somehow
    best_bias = np.asarray(best_bias)
    best_bias.sort(axis=-1)
    best_noise = np.asarray(best_noise)
    best_noise.sort(axis=-1)
    return best_bias, best_noise


class MergeStrategy(enum.Enum):
    MAX = "max"
    MIDDLE = "middle"
    MEAN = "mean"


def make_thresholds(
    in_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    radius: int = 3,
    n_best: int = 3,
    merge_strategy: MergeStrategy = MergeStrategy.MAX,
    skip_spatial: bool = False,
    bias_scale = 3.5,
    noise_scale = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attempt to compute bias / noise thresholds without domain knowledge

    Calls the lower level `estimate_bias_and_noise` function and applies some filtering 
    and heurstics.
    """
    bias, noise = estimate_bias_and_noise(in_data, mask, radius, n_best)
    if n_best > 1:
        if merge_strategy in (MergeStrategy.MAX, MergeStrategy.MIDDLE):
            if merge_strategy == MergeStrategy.MAX:
                idx = -1
            else:
                idx = n_best // 2
            bias = bias[..., idx]
            noise = noise[..., idx]
        else:
            assert merge_strategy == MergeStrategy.MEAN
            bias = np.mean(bias, axis=-1)
            noise = np.mean(noise, axis=-1)
    if mask is not None:
        bias[mask == 0] = robust_mean(bias)
        noise[mask == 0] = robust_mean(noise)
    # Reduce outliers with spatial filtering
    if not skip_spatial:
        bias = gaussian_filter(median_filter(bias), 1.0)
        noise = gaussian_filter(median_filter(noise), 1.0)
    return bias * bias_scale, noise * noise_scale 


accum_dtype_map = {
    np.dtype(np.uint16): np.float32,
    np.dtype(np.uint32): np.float64,
    np.dtype(np.int16): np.float32,
    np.dtype(np.int32): np.float64,
    np.dtype(np.float32): np.float64,
    np.dtype(np.float64): np.float64,
}


def denoise_tnlm(
    in_data: np.ndarray,
    bias_thresh: Optional[np.ndarray] = None,
    noise_thresh: Optional[np.ndarray] = None,
    radius: int = 5,
    mask: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
):
    """Apply temporal nl-means denoising to a 4D array
    
    If the thresholds `bias_thresh` and `noise_thresh` aren't given they will be created
    via the `make_thresholds` function with default arguments.
    """
    if mask is None:
        mask = np.ones(in_data.shape[:3], dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8, copy=False)
    if bias_thresh is None or noise_thresh is None:
        threshes = make_thresholds(in_data, mask)
        if bias_thresh is None:
            bias_thresh = threshes[0]
        if noise_thresh is None:
            noise_thresh = threshes[1]
        del threshes
    if out is None:
        out = np.zeros_like(in_data)
    elif out.shape != in_data.shape:
        raise ValueError("Provided 'out' array is wrong shape")
    accum_dtype = accum_dtype_map[in_data.dtype]
    accum_buf = np.empty(in_data.shape[-1], accum_dtype_map[in_data.dtype])
    log.debug(f"Using {accum_dtype=} for accumulation buffer")
    return np.asarray(
        temporal_nl_means(
            in_data, bias_thresh, noise_thresh, mask, out, accum_buf, radius
        )
    )


def _setup_app_logging(verbose, debug):
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


error_console = Console(stderr=True, style="bold red")


thresh_app = typer.Typer()


@thresh_app.command("tnlm_thresh")
def make_thresholds_files(
    in_path: Path,
    radius: int = 5,
    mask: Optional[Path] = None,
    local_top: int = 3,
    merge_strategy: MergeStrategy = MergeStrategy.MAX,
    out_dir: Optional[Path] = None,
    skip_spatial: bool = False,
    bias_scale: float = 3.5,
    noise_scale: float = 7.0,
    verbose: bool = False,
    debug: bool = False,
):
    """Estimate bias / noise thresholds from neighborhood voxels"""
    _setup_app_logging(verbose, debug)
    if out_dir is None:
        out_dir = Path(".")
    else:
        if not out_dir.exists():
            error_console.print(f"Output directory doesn't exist: {out_dir}")
            return 1
        elif not out_dir.is_dir():
            error_console.print(f"The given 'out_dir' isn't a directory: {out_dir}")
            return 1
    in_img = nb.load(in_path)
    in_data = np.asanyarray(in_img.dataobj)
    if mask is not None:
        mask = np.asarray(nb.load(mask).dataobj)
    bias_thresh, noise_thresh = make_thresholds(
        in_data, 
        mask, 
        radius, 
        local_top, 
        merge_strategy, 
        skip_spatial, 
        bias_scale, 
        noise_scale
    )
    del in_data
    bias_nii = nb.Nifti1Image(bias_thresh, in_img.affine)
    nb.save(bias_nii, out_dir / "tnlm_bias_thresh.nii.gz")
    del bias_nii, bias_thresh
    noise_nii = nb.Nifti1Image(noise_thresh, in_img.affine)
    nb.save(noise_nii, out_dir / "tnlm_noise_thresh.nii.gz")


denoise_app = typer.Typer()


@denoise_app.command("tnlm_denoise")
def denoise_tnlm_files(
    in_path: Path,
    bias_thresh: Optional[Path] = None,
    noise_thresh: Optional[Path] = None,
    radius: int = 5,
    mask: Optional[Path] = None,
    out_path: Optional[Path] = None,
    save_thresholds: bool = False,
    verbose: bool = False,
    debug: bool = False,
):
    """Apply temporal nl-means denoising to a 4D image file"""
    _setup_app_logging(verbose, debug)
    # Load data, process, save results
    in_img = nb.load(in_path)
    in_data = np.asanyarray(in_img.dataobj)
    if mask is not None:
        mask = np.asarray(nb.load(mask).dataobj)
    if out_path is None:
        out_path = in_path.parent / f"denoised_{in_path.name}"
    if bias_thresh is None or noise_thresh is None:
        if not bias_thresh is None or not noise_thresh is None:
            error_console("Either provide both thresholds or none") 
        log.info("Computing thresholds...")
        bias_thresh, noise_thresh = make_thresholds(in_data, mask)
        if save_thresholds:
            bias_nii = nb.Nifti1Image(bias_thresh, in_img.affine)
            nb.save(bias_nii, out_path.parent / f"bias_thresh_{in_path.name}")
            del bias_nii
            noise_nii = nb.Nifti1Image(noise_thresh, in_img.affine)
            nb.save(noise_nii, out_path.parent / f"noise_thresh_{in_path.name}")
            del noise_nii
    log.info("Starting denoising...")
    out_data = denoise_tnlm(in_data, bias_thresh, noise_thresh, radius, mask)
    log.info("Done")
    out_img = nb.Nifti1Image(out_data, in_img.affine)
    nb.save(out_img, out_path)
