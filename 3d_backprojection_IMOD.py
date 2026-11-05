#!/usr/bin/env python3
"""Align and combine *_3dmontage.mrc volumes using IMOD translations.

Python translation of Backproject_align3d_IMOD.m.

Required packages:
    pip install numpy scipy mrcfile scikit-image

Example:
    python 3d_backprojection_IMOD.py \
        --first-s 0 --end-s 40 --step-s 1 \
        --first-3dmontage 261014_AD200_tomo8_s0_3dmontage.mrc \
        --rawtlt 261014_AD200_tomo8.rawtlt \
        --xf 261014_AD200_tomo8.xf \
        --background-padding 0

Array convention:
    Internally volumes use the MATLAB-like order (X, Y, Z). MRC files are read
    and written in mrcfile order (Z, Y, X), with an explicit transpose at the
    boundary. Except for translation, operations retain the argument/dimension
    order used by the MATLAB source.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift
from skimage.transform import resize


def imtranslate_xy(image: np.ndarray, dx: float, dy: float,
                   order: int = 1, fill_value: float = 0.0) -> np.ndarray:
    """Translate with the requested Python convention: shift=(dx, dy)."""
    return nd_shift(
        image,
        shift=(float(dx), float(dy)),
        order=order,
        mode="constant",
        cval=float(fill_value),
        prefilter=(order > 1),
    )


def shift_to_center(image: np.ndarray, Xc: float, Yc: float,
                    order: int = 1, fill_value: float = 0.0) -> np.ndarray:
    """Shift a feature at (Xc, Yc) to the origin."""
    return imtranslate_xy(
        image, dx=-float(Xc), dy=-float(Yc),
        order=order, fill_value=fill_value,
    )


def compose_shifts(*pairs: tuple[float, float]) -> tuple[float, float]:
    """Vector-add multiple (dx, dy) shifts."""
    return sum(pair[0] for pair in pairs), sum(pair[1] for pair in pairs)


def read_mrc_matlab_order(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read MRC (Z,Y,X) and return float64 volume in MATLAB-like (X,Y,Z)."""
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float64)
        voxel = (
            float(mrc.voxel_size.x),
            float(mrc.voxel_size.y),
            float(mrc.voxel_size.z),
        )
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D MRC volume in {path}, got {data.shape}.")
    return np.transpose(data, (2, 1, 0)), voxel


def write_mrc_matlab_order(path: Path, volume: np.ndarray,
                           voxel_size: tuple[float, float, float]) -> None:
    """Write MATLAB-like (X,Y,Z) volume to MRC order (Z,Y,X)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mrc_data = np.transpose(volume, (2, 1, 0)).astype(np.float32, copy=False)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(mrc_data)
        if all(np.isfinite(voxel_size)) and all(value > 0 for value in voxel_size):
            mrc.voxel_size = voxel_size
        mrc.update_header_stats()


def resize_bicubic_matlab(image: np.ndarray, shape: tuple[int, int], *, antialias: bool) -> np.ndarray:
    """Resize similarly to MATLAB imresize(..., 'bicubic').

    MATLAB applies antialiasing when shrinking.  That step is essential here
    because the normal background estimate shrinks each axis to 2% before
    restoring the original size.
    """
    target = (max(1, int(shape[0])), max(1, int(shape[1])))
    image = np.asarray(image, dtype=np.float64)
    if image.shape == target:
        return image.copy()

    return resize(
        image,
        target,
        order=3,
        mode="edge",
        cval=0.0,
        clip=False,
        preserve_range=True,
        anti_aliasing=bool(antialias),
    ).astype(np.float64, copy=False)


def matlab_round_positive(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def matlab_background_subtract(cropped: np.ndarray) -> np.ndarray:
    """MATLAB normal path: imresize to 2%, restore, then subtract.

    The reduction is antialiased, as MATLAB imresize does by default when the
    scale is below one.  The former optional 5%/extreme path was removed.
    """
    fraction = 0.02
    small_shape = (
        max(1, matlab_round_positive(fraction * cropped.shape[0])),
        max(1, matlab_round_positive(fraction * cropped.shape[1])),
    )
    local = resize_bicubic_matlab(cropped, small_shape, antialias=True)
    local = resize_bicubic_matlab(local, cropped.shape, antialias=False)
    return np.asarray(cropped, dtype=np.float64) - local


def positive_crop_bounds(image: np.ndarray, top_margin: int,
                         bottom_margin: int, left_margin: int,
                         right_margin: int) -> tuple[int, int, int, int] | None:
    """Return Python-exclusive bounds matching MATLAB min/max positive-pixel crops."""
    rows, cols = np.nonzero(image > 0)
    if rows.size == 0 or float(np.sum(image)) < 10.0:
        return None
    top = int(rows.min()) + top_margin
    bottom = int(rows.max()) - bottom_margin + 1
    left = int(cols.min()) + left_margin
    right = int(cols.max()) - right_margin + 1
    if top >= bottom or left >= right:
        return None
    return top, bottom, left, right


def load_rawtlt(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def load_xf(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] < 6:
        raise ValueError(f"IMOD XF file must have at least 6 columns: {path}")
    return values


def derive_montage_path(first_path: Path, first_s: int, snumber: int) -> Path:
    """Replace the s-index in the selected first *_3dmontage.mrc filename."""
    name = first_path.name
    patterns = [
        rf"(?i)_s{re.escape(str(first_s))}_3dmontage\.mrc$",
        rf"(?i)_s{re.escape(str(first_s))}\.mrc$",
    ]
    replacements = [f"_s{snumber}_3dmontage.mrc", f"_s{snumber}.mrc"]
    for pattern, replacement in zip(patterns, replacements):
        if re.search(pattern, name):
            return first_path.with_name(re.sub(pattern, replacement, name))
    raise ValueError(
        f"First montage filename {name!r} does not end with "
        f"'_s{first_s}_3dmontage.mrc' or '_S{first_s}.mrc'."
    )


def derive_output_prefix(first_path: Path, first_s: int) -> Path:
    name = first_path.name
    name = re.sub(
        rf"(?i)_s{re.escape(str(first_s))}_3dmontage\.mrc$", "", name
    )
    name = re.sub(rf"(?i)_s{re.escape(str(first_s))}\.mrc$", "", name)
    return first_path.with_name(name)


def build_alignment_vectors(actual_angles_order_deg: np.ndarray,
                            xf_table: np.ndarray,
                            snumber_vec0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the MATLAB rawtlt/XF permutation and translation extraction."""
    if xf_table.shape[0] != actual_angles_order_deg.size:
        raise ValueError(
            f"XF rows ({xf_table.shape[0]}) differ from rawtlt angle count "
            f"({actual_angles_order_deg.size})."
        )
    sorted_positions = np.argsort(actual_angles_order_deg, kind="stable")
    inverse_positions = np.argsort(sorted_positions, kind="stable")
    xf_rows_for_acquisition = inverse_positions[snumber_vec0]

    # MATLAB: alignX = Var6; alignY = Var5 / cos(angle).
    align_x = -xf_table[xf_rows_for_acquisition, 5].astype(np.float64)
    cosine = np.cos(np.deg2rad(actual_angles_order_deg[snumber_vec0]))
    if np.any(np.abs(cosine) < 1e-12):
        raise ValueError("Cannot divide the IMOD Y shift by cos(angle) near 90 degrees.")
    align_y = xf_table[xf_rows_for_acquisition, 4].astype(np.float64) / cosine
    align_x = np.nan_to_num(align_x, nan=0.0)
    align_y = np.nan_to_num(align_y, nan=0.0)
    return align_x, align_y


def reconstruct(args: argparse.Namespace) -> tuple[Path, Path]:
    snumber_vec0 = np.arange(
        args.first_s, args.end_s + (1 if args.step_s > 0 else -1), args.step_s,
        dtype=np.int64,
    )
    if snumber_vec0.size == 0:
        raise ValueError("The first/end/step S arguments produce an empty selection.")
    if np.any(snumber_vec0 < 0):
        raise ValueError("S indices must be zero or positive.")

    angles = load_rawtlt(args.rawtlt)
    if int(snumber_vec0.max()) >= angles.size:
        raise IndexError(
            f"Largest selected S index {snumber_vec0.max()} exceeds rawtlt "
            f"length {angles.size}."
        )
    selected_angles = angles[snumber_vec0]
    xf_table = load_xf(args.xf)
    align_x, align_y = build_alignment_vectors(angles, xf_table, snumber_vec0)

    prefix = derive_output_prefix(args.first_3dmontage, args.first_s)
    rec_out = Path(str(prefix) + "_IMODAlign3dRec.mrc")
    center_out = Path(str(prefix) + "_ATcenterRecs.mrc")

    grand_sum = grand_count = stack_center = None
    reference_shape = None
    voxel_size = (1.0, 1.0, 1.0)
    valid_tilts = 0

    for selection_index, (snumber, angle, shift_x, shift_y) in enumerate(
        zip(snumber_vec0, selected_angles, align_x, align_y), start=1
    ):
        path = derive_montage_path(args.first_3dmontage, args.first_s, int(snumber))
        print(
            f"[{selection_index}/{len(snumber_vec0)}] s{snumber}, "
            f"angle={angle:g}, shiftX={shift_x:g}, shiftY={shift_y:g}: {path.name}",
            flush=True,
        )
        if not path.is_file():
            print("  Missing; skipped.", flush=True)
            continue

        rec1, current_voxel = read_mrc_matlab_order(path)
        if reference_shape is None:
            reference_shape = rec1.shape
            voxel_size = current_voxel
            grand_sum = np.zeros(reference_shape, dtype=np.float64)
            grand_count = np.zeros(reference_shape, dtype=np.float64)
            stack_center = np.zeros(
                (reference_shape[0], reference_shape[1], len(snumber_vec0)),
                dtype=np.float32,
            )
        elif rec1.shape != reference_shape:
            raise ValueError(
                f"Volume {path} has shape {rec1.shape}; expected {reference_shape}."
            )

        num_z = rec1.shape[2]
        # MATLAB uses floor(numz/2) with one-based indexing. Convert to Python.
        center_z = max(0, int(math.floor(num_z / 2.0)) - 1)
        center = rec1[:, :, center_z]
        bounds = positive_crop_bounds(center, 3, 3, 3, 3)
        if bounds is None:
            print("  Empty/weak center plane; skipped.", flush=True)
            continue
        top, bottom, left, right = bounds
        center_crop = center[top:bottom, left:right]
        center_hp = matlab_background_subtract(center_crop)

        center_clean = np.full(center.shape, args.background_padding, dtype=np.float64)
        center_clean[top:bottom, left:right] = center_hp

        # MATLAB calls imtranslate(image, [-shiftsY -shiftsX]). The requested
        # Python convention reverses that command to dx=-shiftsX, dy=-shiftsY.
        center_shifted = imtranslate_xy(
            center_clean,
            dx=-float(shift_x),
            dy=-float(shift_y),
            order=args.shift_order,
            fill_value=args.background_padding,
        )
        stack_center[:, :, selection_index - 1] = center_shifted.astype(np.float32)

        rec_cor = np.zeros(reference_shape, dtype=np.float64)
        rec_mask = np.zeros(reference_shape, dtype=np.float64)

        for z_index in range(num_z):
            image = rec1[:, :, z_index]
            z_bounds = positive_crop_bounds(image, 5, 1, 5, 1)
            if z_bounds is None:
                continue
            z_top, z_bottom, z_left, z_right = z_bounds
            crop = image[z_top:z_bottom, z_left:z_right]
            hp = matlab_background_subtract(crop)

            # MATLAB imgaussfilt(...,7), then remove 10 pixels on each side.
            evaluated = gaussian_filter(hp, sigma=7.0, mode="nearest")
            if evaluated.shape[0] > 20 and evaluated.shape[1] > 20:
                evaluated_inner = evaluated[10:-10, 10:-10]
            else:
                evaluated_inner = evaluated
            normalization = float(np.std(evaluated_inner, ddof=1))
            if not np.isfinite(normalization) or normalization <= args.min_normalization:
                print(
                    f"  z={z_index}: invalid normalization {normalization}; skipped.",
                    flush=True,
                )
                continue

            clean = np.full(center.shape, args.background_padding, dtype=np.float64)
            clean[z_top:z_bottom, z_left:z_right] = hp
            mask = np.zeros(center.shape, dtype=np.float64)
            mask[z_top:z_bottom, z_left:z_right] = 1.0

            shifted = imtranslate_xy(
                clean,
                dx=-float(shift_x),
                dy=-float(shift_y),
                order=args.shift_order,
                fill_value=args.background_padding,
            )
            shifted_mask = imtranslate_xy(
                mask,
                dx=-float(shift_x),
                dy=-float(shift_y),
                order=0,
                fill_value=0.0,
            ) > 0
            rec_cor[:, :, z_index] = shifted / normalization
            rec_mask[:, :, z_index] = shifted_mask.astype(np.float64)

        grand_sum += rec_cor
        grand_count += rec_mask
        valid_tilts += 1

    if reference_shape is None or valid_tilts == 0:
        raise RuntimeError("No valid 3dmontage volume was processed.")

    grand_rec = np.divide(
        grand_sum,
        grand_count,
        out=np.zeros_like(grand_sum),
        where=grand_count > 0,
    )
    valid = grand_count > 0
    mean_rec = float(np.mean(grand_rec[valid])) if np.any(valid) else 0.0
    grand_rec[~valid] = mean_rec

    write_mrc_matlab_order(center_out, stack_center, voxel_size)
    write_mrc_matlab_order(rec_out, grand_rec, voxel_size)
    print(f"Wrote center stack: {center_out}", flush=True)
    print(f"Wrote reconstruction: {rec_out}", flush=True)
    return rec_out, center_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter, crop, translate, normalize, and combine *_3dmontage.mrc "
            "volumes using translation-only IMOD alignments."
        )
    )
    parser.add_argument("--first-s", type=int, required=True,
                        help="First S index in acquisition order.")
    parser.add_argument("--end-s", type=int, required=True,
                        help="Final S index in acquisition order, inclusive.")
    parser.add_argument("--step-s", type=int, required=True,
                        help="Step between selected S indices.")
    parser.add_argument("--first-3dmontage", type=Path, required=True,
                        help="First selected *_s#_3dmontage.mrc file.")
    parser.add_argument("--rawtlt", type=Path, required=True,
                        help="Raw tilt-angle file in acquisition order.")
    parser.add_argument("--xf", type=Path, required=True,
                        help="IMOD .xf file ordered by increasing tilt angle.")
    parser.add_argument("--background-padding", type=float, default=0.0,
                        help="Fill value used outside shifted/cropped data (default: 0).")
    parser.add_argument("--shift-order", type=int, choices=(0, 1, 3), default=1,
                        help="Interpolation order for image translations (default: 1).")
    parser.add_argument("--min-normalization", type=float, default=1e-12,
                        help="Reject planes with a smaller/invalid normalization.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.step_s == 0:
        raise ValueError("--step-s cannot be zero.")
    reconstruct(args)


if __name__ == "__main__":
    main()
