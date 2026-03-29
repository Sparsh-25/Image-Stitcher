import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# A1 — Exposure Matching
# ============================================================================
# WHY: After warping Image A into Image B's frame, both images may have
# different overall brightness/white-balance because they were taken with
# slightly different auto-exposure settings. Even perfect geometry produces
# a visible seam if the pixel values on either side don't match.
#
# HOW (Histogram Matching):
#   1. Work in YCrCb colorspace so we only adjust luminance (Y channel),
#      leaving the colours (Cr, Cb) unchanged.
#   2. Compute the Cumulative Distribution Function (CDF) of Y in both images.
#   3. Build a 256-entry LUT: for each intensity value i in src,
#      find the intensity j in ref whose CDF(j) == CDF_src(i).
#      This is called "histogram specification" or "histogram matching".
#   4. Apply the LUT to src's Y channel.
# ============================================================================

def match_exposure(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Match the luminance histogram of `src` to that of `ref`.

    Operates only on the Y (luminance) channel in YCrCb space, so
    colours (Cr, Cb) are not affected — only overall brightness is adjusted.

    Args:
        src: BGR image to adjust (Image A, warped onto canvas).
        ref: BGR reference image whose brightness to match (Image B).

    Returns:
        BGR image with src's brightness matched to ref. Same shape as src.
    """
    # Convert both to YCrCb.
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

    src_y = src_ycrcb[:, :, 0]   # Luminance channel of src
    ref_y = ref_ycrcb[:, :, 0]   # Luminance channel of ref

    # Compute normalised CDF for each image's Y channel.
    # np.bincount counts how many pixels have each intensity value (0–255).
    src_hist = np.bincount(src_y.ravel(), minlength=256).astype(np.float64)
    ref_hist = np.bincount(ref_y.ravel(), minlength=256).astype(np.float64)

    # Normalise to get probability mass functions, then cumsum for CDF.
    src_cdf = (src_hist / src_hist.sum()).cumsum()
    ref_cdf = (ref_hist / ref_hist.sum()).cumsum()

    # Build LUT: for each intensity i in src, find the intensity j in ref
    # whose CDF value is closest. np.searchsorted finds this efficiently.
    lut = np.searchsorted(ref_cdf, src_cdf).astype(np.uint8)  # shape (256,)

    # Apply LUT to src's Y channel only.
    matched_y = lut[src_y]

    # Rebuild YCrCb image with the matched Y channel.
    result_ycrcb = src_ycrcb.copy()
    result_ycrcb[:, :, 0] = matched_y

    return cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)


# ============================================================================
# A2 — Gradient Blend Mask
# ============================================================================
# WHY: The Laplacian pyramid needs a float32 mask that tells it how much of
# each image to take at each pixel.
#   mask = 0.0  → show 100% Image A
#   mask = 1.0  → show 100% Image B
#   mask = 0.5  → 50/50 blend
#
# WHY GRADIENT (not binary 0/1)?
# A hard 0→1 step at the seam boundary is still a sharp edge at the finest
# pyramid level. The pyramid blurs it at coarser levels, but Level 0 (full
# resolution) still sees the jump → visible seam line.
# A gradient means even Level 0 gets a smooth function → seam invisible.
#
# WHY NUMPY (not for-loop)?
# The canvas can be 4000–6000 px wide. A Python for-loop over columns
# takes 4–5 seconds. NumPy broadcasting does it in under 10 ms.
# ============================================================================

def build_blend_mask(
    canvas_shape: tuple,
    warped_a: np.ndarray,
    canvas_b: np.ndarray,
    seam_band_px: int = 0,
) -> np.ndarray:
    """
    Build a float32 blend mask for the panorama canvas.

    Finds the optimal seam column (minimum mean absolute difference over the
    overlap zone, fully vectorised) and places a narrow gradient there.
    seam_band_px=0 gives a hard binary cut at the optimal column.

    M = 0 in A-only zone, 1 in B-only zone.
    Direction (gradient 0->1 or 1->0) is auto-detected.
    """
    H, W = canvas_shape[:2]

    gray_a = cv2.cvtColor(warped_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(canvas_b,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    has_a  = gray_a > 0
    has_b  = gray_b > 0

    # Base mask: 1 everywhere B has content, 0 elsewhere.
    mask = has_b.astype(np.float32)

    both_col     = (has_a & has_b).any(axis=0)
    overlap_cols = np.where(both_col)[0]
    if len(overlap_cols) == 0:
        return mask

    col_left  = int(overlap_cols[0])
    col_right = int(overlap_cols[-1])

    # Vectorised optimal seam: per-column mean abs-diff over overlap rows.
    both_2d = has_a & has_b                        # (H, W) bool
    diff    = np.abs(gray_a - gray_b)              # (H, W)
    # Row-count and diff-sum per column (vectorised, no Python loop).
    n_rows  = both_2d.sum(axis=0).astype(np.float32)     # (W,)
    sum_diff= (diff * both_2d).sum(axis=0)               # (W,)
    col_score = np.where(n_rows > 0, sum_diff / np.maximum(n_rows, 1), 1e9)

    seam_col = int(col_left + np.argmin(col_score[col_left:col_right + 1]))

    # Detect whether B is to the RIGHT or LEFT of A.
    a_cols   = np.where(has_a.any(axis=0))[0]
    b_cols   = np.where(has_b.any(axis=0))[0]
    a_center = int(a_cols.mean()) if len(a_cols) else W // 2
    b_center = int(b_cols.mean()) if len(b_cols) else W // 2

    half_band  = seam_band_px // 2
    band_left  = max(0,   seam_col - half_band)
    band_right = min(W-1, seam_col + half_band)
    band_w     = max(band_right - band_left, 1)

    col_idx = np.arange(W, dtype=np.float32)
    if b_center >= a_center:
        gradient = np.clip((col_idx - band_left) / band_w, 0.0, 1.0)
    else:
        gradient = np.clip((band_right - col_idx) / band_w, 0.0, 1.0)

    # Apply gradient only inside the seam band within the overlap zone.
    in_band = (col_idx >= band_left) & (col_idx <= band_right)
    in_seam = in_band[np.newaxis, :] * both_2d

    mask = mask * (1.0 - in_seam) + gradient[np.newaxis, :] * in_seam
    return mask.astype(np.float32)

# ============================================================================
# Quick demo — run as: python laplacian_blender2.py  (A1 + A2)
# ============================================================================


# ============================================================================
# A3 - Laplacian Pyramid Blend
# ============================================================================
# HOW IT WORKS:
#   A Laplacian pyramid decomposes an image into frequency bands:
#     L0 = finest detail (high frequency: edges, textures)
#     L1 = medium structures
#     L2..LN = coarse colour/brightness (low frequency)
#
#   We blend each band with a correspondingly blurred version of the mask:
#     - At coarse levels (LN), the mask is heavily blurred -> wide colour transition
#     - At fine levels (L0), the mask is narrow -> sharp detail preserved
#
#   Collapsing the blended pyramid back gives a seamless result at all scales.
#
# THE pyrUp RESIZE FIX:
#   cv2.pyrDown then cv2.pyrUp is NOT a round-trip for odd-sized dimensions.
#   A 4733px image pyrDown -> 2367px -> pyrUp -> 4734px (off by 1).
#   This causes shape mismatch when computing L_k = G_k - pyrUp(G_{k+1}).
#   Fix: after every pyrUp, explicitly resize to match G_k's exact dimensions.
# ============================================================================

class LaplacianBlender:
    """
    Seam-free image blending using a Laplacian pyramid.

    Splits both images into frequency bands, blends each band with a
    progressively blurred version of the gradient mask, then collapses
    the pyramid back to full resolution.
    """

    def __init__(self, num_levels: int = 6):
        """
        Args:
            num_levels: Number of pyramid levels.
                        6 works well for images with 500-5000px overlap zones.
                        Fewer levels -> seam still visible at coarse scale.
                        More levels -> diminishing returns past 8.
        """
        self.num_levels = num_levels

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_gaussian_pyramid(self, img: np.ndarray) -> list:
        """
        Build a Gaussian pyramid by repeatedly downsampling.
        Returns list [G0, G1, ..., G_{num_levels}] — G0 is the original.
        """
        pyramid = [img.astype(np.float32)]
        for _ in range(self.num_levels):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid

    def _build_laplacian_pyramid(self, gauss_pyr: list) -> list:
        """
        Build Laplacian pyramid from a Gaussian pyramid.
        L_k = G_k - pyrUp(G_{k+1})  resized to G_k's exact shape.
        Last level = G_{num_levels} (no subtraction needed).
        """
        laplacian = []
        for k in range(self.num_levels):
            g_k    = gauss_pyr[k]
            g_next = gauss_pyr[k + 1]
            # Upsample and resize to exactly match g_k's dimensions.
            g_up   = cv2.pyrUp(g_next)
            g_up   = cv2.resize(g_up, (g_k.shape[1], g_k.shape[0]))
            laplacian.append(g_k - g_up)
        laplacian.append(gauss_pyr[self.num_levels])   # coarsest level
        return laplacian

    def _collapse_pyramid(self, laplacian_pyr: list) -> np.ndarray:
        """
        Reconstruct full-resolution image by collapsing the Laplacian pyramid.
        Start from coarsest level and iteratively add finer detail.
        """
        result = laplacian_pyr[-1]
        for k in range(self.num_levels - 1, -1, -1):
            result = cv2.pyrUp(result)
            result = cv2.resize(result, (laplacian_pyr[k].shape[1],
                                         laplacian_pyr[k].shape[0]))
            result = result + laplacian_pyr[k]
        return result

    # ── Public API ─────────────────────────────────────────────────────────

    def blend(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        mask:  np.ndarray
    ) -> np.ndarray:
        """
        Blend img_a and img_b using the Laplacian pyramid.

        Args:
            img_a:  First image (BGR uint8 or float32), placed on canvas.
            img_b:  Second image (BGR uint8 or float32), placed on canvas.
            mask:   float32 (H, W) blend mask — 0=img_a, 1=img_b, gradient in overlap.

        Returns:
            Blended BGR uint8 image of same shape as img_a / img_b.
        """
        # Ensure float32 inputs for the pyramid arithmetic.
        a = img_a.astype(np.float32)
        b = img_b.astype(np.float32)

        # Keep mask 2D (H, W) — pyrDown handles 2D arrays correctly.
        # We broadcast the channel dim at blend time with [:,:,np.newaxis].
        m = mask.astype(np.float32)

        # Build Gaussian pyramids for all three.
        gauss_a = self._build_gaussian_pyramid(a)
        gauss_b = self._build_gaussian_pyramid(b)
        gauss_m = self._build_gaussian_pyramid(m)

        # Build Laplacian pyramids for both images.
        lap_a = self._build_laplacian_pyramid(gauss_a)
        lap_b = self._build_laplacian_pyramid(gauss_b)

        # Blend each pyramid level:
        #   blended[k] = M[k] * L_b[k]  +  (1 - M[k]) * L_a[k]
        # Expand mask dim from (H,W) -> (H,W,1) so it broadcasts against (H,W,3).
        blended_pyr = []
        for k in range(self.num_levels + 1):
            mk = gauss_m[k][:, :, np.newaxis]   # (h,w,1) broadcasts with (h,w,3)
            bk = mk * lap_b[k] + (1.0 - mk) * lap_a[k]
            blended_pyr.append(bk)

        # Collapse pyramid back to full resolution.
        result = self._collapse_pyramid(blended_pyr)

        # Clip and convert back to uint8.
        return np.clip(result, 0, 255).astype(np.uint8)

# ============================================================================
# Standalone Demo  (run as: python laplacian_blender2.py)
# ============================================================================

if __name__ == "__main__":
    import sys
    from feature_matcher1 import FeatureMatcher
    from homography_estimator1 import HomographyEstimator, print_homography_report
    from image_stitcher1 import ImageStitcher

    PATH_A = "/Users/Antino/Desktop/Image Stitcher/images/left_room.jpg"
    PATH_B = "/Users/Antino/Desktop/Image Stitcher/images/right_room.jpg"

    img_a = cv2.imread(PATH_A)
    img_b = cv2.imread(PATH_B)
    if img_a is None:
        print(f"[ERROR] Could not load: {PATH_A}"); sys.exit(1)
    if img_b is None:
        print(f"[ERROR] Could not load: {PATH_B}"); sys.exit(1)

    print("[Step 1] SIFT feature matching...")
    fm = FeatureMatcher(detector_type="SIFT", ratio_threshold=0.7)
    kps_a, desc_a = fm.detect_and_describe(img_a)
    kps_b, desc_b = fm.detect_and_describe(img_b)
    good = fm.match(desc_a, desc_b)
    pts_a, pts_b = fm.extract_point_pairs(kps_a, kps_b, good)
    print(f"         Good matches: {len(good)}")

    print("[Step 2] Homography estimation (MAGSAC++)...")
    est = HomographyEstimator()
    H, hmask = est.estimate(pts_a, pts_b)
    stats = est.compute_reprojection_errors(pts_a, pts_b, H, hmask)
    print_homography_report(H, stats)

    print("[Step 3] Building canvas...")
    stitcher = ImageStitcher()
    h_b, w_b = img_b.shape[:2]
    canvas_w, canvas_h, T, off_x, off_y = stitcher._compute_canvas_size(img_a, img_b, H)

    print("[A1] Exposure matching...")
    img_a_matched = match_exposure(src=img_a, ref=img_b)
    TH = T @ H
    warped_a = cv2.warpPerspective(img_a_matched, TH, (canvas_w, canvas_h))
    canvas_b = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_b[off_y:off_y + h_b, off_x:off_x + w_b] = img_b

    print("[A2] Building gradient blend mask...")
    blend_mask = build_blend_mask(
        canvas_shape=(canvas_h, canvas_w),
        warped_a=warped_a,
        canvas_b=canvas_b
    )

    print("[A3] Laplacian pyramid blend (6 levels)...")
    blender  = LaplacianBlender(num_levels=6)
    panorama = blender.blend(warped_a, canvas_b, blend_mask)

    out_path = "/Users/Antino/Desktop/Image Stitcher/panorama_blended.jpg"
    cv2.imwrite(out_path, panorama)
    print(f"[OK] panorama_blended.jpg saved  ({panorama.shape[1]} x {panorama.shape[0]} px)")
    print("[A4] Feature A complete. panorama_blended.jpg is the seam-free stitched output.")
