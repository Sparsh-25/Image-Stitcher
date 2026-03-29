import cv2
import numpy as np
import os

from feature_matcher1 import FeatureMatcher
from homography_estimator1 import HomographyEstimator, print_homography_report
from image_stitcher1 import ImageStitcher
from laplacian_blender2 import match_exposure, build_blend_mask, LaplacianBlender


# ============================================================================
# B1 — Load N Images in Order
# ============================================================================
# Loads a list of image file paths left-to-right, validates each one,
# and returns a list of BGR arrays. Exits loudly if any file is missing
# or corrupt — silent None from cv2.imread is a common hidden bug.
# ============================================================================

def load_images(paths: list) -> list:
    """
    Load and validate a list of image paths.

    Args:
        paths: Ordered list of file paths (left to right).

    Returns:
        List of BGR uint8 arrays, same order as paths.

    Raises:
        SystemExit if any image fails to load.
    """
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Could not load image: {path}")
            raise SystemExit(1)
        h, w = img.shape[:2]
        print(f"   Loaded: {os.path.basename(path)}  ({w} x {h} px)")
        images.append(img)
    return images


# ============================================================================
# B2 — Pairwise Homography Chain
# ============================================================================
# For N images, compute N-1 pairwise homographies:
#   H[0] maps img[0] -> img[1]
#   H[1] maps img[1] -> img[2]
#   ...
#   H[k] maps img[k] -> img[k+1]
#
# Each pair goes through SIFT + Lowe's Ratio Test + MAGSAC++.
# ============================================================================

def compute_pairwise_homographies(
    images: list,
    ratio_threshold: float = 0.7,
    max_reproj_error: float = 5.0
) -> list:
    """
    Compute pairwise homographies for consecutive image pairs.

    Args:
        images:           Ordered list of BGR images.
        ratio_threshold:  Lowe's Ratio Test threshold.
        max_reproj_error: MAGSAC++ reprojection error threshold (pixels).

    Returns:
        List of (N-1) 3x3 homography matrices. H[k] maps images[k] -> images[k+1].
    """
    fm  = FeatureMatcher(detector_type="SIFT", ratio_threshold=ratio_threshold)
    est = HomographyEstimator(max_reproj_error=max_reproj_error)

    homographies = []
    for k in range(len(images) - 1):
        print(f"\n[B2] Pair ({k},{k+1}):")

        kps_a, desc_a = fm.detect_and_describe(images[k])
        kps_b, desc_b = fm.detect_and_describe(images[k + 1])
        good_matches  = fm.match(desc_a, desc_b)

        if len(good_matches) < 4:
            raise ValueError(
                f"Only {len(good_matches)} matches for pair ({k},{k+1}). "
                "Increase overlap between images or lower ratio_threshold."
            )

        pts_a, pts_b = fm.extract_point_pairs(kps_a, kps_b, good_matches)
        H, mask      = est.estimate(pts_a, pts_b)
        stats        = est.compute_reprojection_errors(pts_a, pts_b, H, mask)
        print_homography_report(H, stats)

        homographies.append(H)

    return homographies


# ============================================================================
# B3 — Global Canvas Computation
# ============================================================================
# Chain the pairwise homographies to get the global transform for each image
# relative to image[0]'s coordinate frame (anchored at image 0).
#
# H_global[0] = Identity  (image 0 stays put)
# H_global[1] = H[0]
# H_global[2] = H[0] @ H[1]   (apply H[0] first, then H[1])
#   ...
# H_global[k] = H[0] @ H[1] @ ... @ H[k-1]
#
# Then project all 4 corners of all N images through their global H,
# compute the union bounding box, and apply a translation offset T so
# no coordinates are negative.
# ============================================================================

def compute_global_canvas(
    images: list,
    homographies: list
) -> tuple:
    """
    Compute global homographies and canvas size for N images.

    Anchors at the MIDDLE image so cumulative scale distortion is symmetric.
    With poor images having 1.3x scale per pair, anchoring at image 0 causes
    1.32^(N-1) cumulative scale on the far image. Center anchoring caps it at
    max 1.32^(N//2) which is significantly smaller.

    H_global[anchor] = Identity
    H_global[k < anchor] = inv(H[k]) @ inv(H[k+1]) @ ... @ inv(H[anchor-1])
    H_global[k > anchor] = H[anchor] @ H[anchor+1] @ ... @ H[k-1]
    """
    N      = len(images)
    anchor = N // 2   # middle image index

    # Build global Hs anchored at the middle image.
    global_Hs = [None] * N
    global_Hs[anchor] = np.eye(3, dtype=np.float64)

    # Forward chain (Images to the RIGHT of anchor): anchor -> anchor+1 -> ... -> N-1
    # To map image k back to anchor, we need the INVERSE of the homography
    for k in range(anchor + 1, N):
        global_Hs[k] = global_Hs[k - 1] @ np.linalg.inv(homographies[k - 1])

    # Backward chain (Images to the LEFT of anchor): anchor -> anchor-1 -> ... -> 0
    # To map image k forward to anchor, we use the STANDARD homography
    for k in range(anchor - 1, -1, -1):
        global_Hs[k] = global_Hs[k + 1] @ homographies[k]

    # Project all corners of all images through their global transforms.
    all_corners = []
    for k, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([
            [0,     0    ],
            [w - 1, 0    ],
            [0,     h - 1],
            [w - 1, h - 1]
        ]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, global_Hs[k])
        all_corners.append(projected)

    all_pts  = np.concatenate(all_corners, axis=0)
    x_min, y_min = all_pts[:, 0, :].min(axis=0)
    x_max, y_max = all_pts[:, 0, :].max(axis=0)

    off_x = max(0.0, -x_min)
    off_y = max(0.0, -y_min)

    canvas_w = int(np.ceil(x_max + off_x)) + 1
    canvas_h = int(np.ceil(y_max + off_y)) + 1

    T = np.array([
        [1, 0, off_x],
        [0, 1, off_y],
        [0, 0, 1    ]
    ], dtype=np.float64)

    return global_Hs, canvas_w, canvas_h, T, int(off_x), int(off_y)


def warp_and_blend_all(
    images: list,
    global_Hs: list,
    canvas_w: int,
    canvas_h: int,
    T: np.ndarray,
    num_blend_levels: int = 3   # kept for API compat but unused
) -> np.ndarray:
    """
    Warp all images onto the global canvas and blend using weighted accumulation.

    Blend model:
        result(x) = sum_i( img_i(x) * w_i(x) ) / sum_i( w_i(x) )

    Weight w_i(x) for each pixel is computed via distance transform from that
    image's valid-pixel boundary. Pixels far from the edge get high weight;
    pixels near the edge get low weight. This gives seamless smooth blending
    at overlaps without any explicit seam mask, and treats all images uniformly
    (no special k==0 case, no order dependency).

    Accumulation is in float32 throughout; uint8 cast only at the end.
    """
    anchor    = len(images) // 2              # center image is exposure reference
    ref_image = images[anchor]

    # Float32 accumulator and weight-sum canvas.
    canvas_f    = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum  = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    for k, img in enumerate(images):
        print(f"[B4] Warping and weighting image {k}...")

        # Exposure-match every image to the center image.
        img_adj = img if k == anchor else match_exposure(src=img, ref=ref_image)

        # Single warp: T @ H_global[k] composed in one call.
        TH = T @ global_Hs[k]
        warped = cv2.warpPerspective(
            img_adj, TH, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Build weight map via distance transform.
        # valid_mask = 1 where warped has non-black content.
        gray        = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        valid_mask  = (gray > 5).astype(np.uint8)
        # Distance of each valid pixel to the nearest invalid (black) pixel.
        dist        = cv2.distanceTransform(valid_mask, cv2.DIST_L2, 5)
        # Normalise to [0, 1] so far-from-edge pixels weight 1.
        d_max       = dist.max()
        if d_max > 0:
            dist = dist / d_max
        weight      = dist[:, :, np.newaxis]   # (H, W, 1) for broadcasting

        # Weighted accumulation in float32 — no uint8 quantization error.
        canvas_f   += warped.astype(np.float32) * weight
        weight_sum += weight

        print(f"   Image {k} accumulated. Valid pixels: {valid_mask.sum():,}")

    # Normalize: divide by sum of weights where any image contributed.
    # Pixels with weight_sum == 0 are pure black (outside all images) → stay 0.
    result = np.where(
        weight_sum > 1e-6,
        canvas_f / np.maximum(weight_sum, 1e-6),
        0.0
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def crop_bounding_box(panorama: np.ndarray) -> np.ndarray:
    """
    Crop panorama to the bounding box of non-black pixels.

    Faster than LIR and retains maximum image area.
    May include small black triangles from warp gaps at corners —
    acceptable for multi-image panoramas where the dominant content is valid.

    Args:
        panorama: BGR uint8 canvas with black padding.

    Returns:
        Cropped BGR uint8 image.
    """
    gray  = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    rows  = np.any(gray > 5, axis=1)   # (H,) — True for rows with content
    cols  = np.any(gray > 5, axis=0)   # (W,) — True for cols with content

    if not rows.any():
        return panorama   # Entirely black — return as-is (shouldn't happen)

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    return panorama[r0:r1 + 1, c0:c1 + 1]


# ============================================================================
# B6 — Standalone Demo
# ============================================================================

if __name__ == "__main__":
    import sys

    # ── Configure your image paths here (left to right order) ────────────
    IMAGE_PATHS = [
        "/Users/Antino/Desktop/Image Stitcher/images/IMG_3620.jpg",
        "/Users/Antino/Desktop/Image Stitcher/images/IMG_3621.jpg",
        "/Users/Antino/Desktop/Image Stitcher/images/IMG_3622.jpg"
    ]
    OUTPUT_PATH = "/Users/Antino/Desktop/Image Stitcher/panorama_multi_new_10.jpg"
    # ─────────────────────────────────────────────────────────────────────

    N = len(IMAGE_PATHS)
    print(f"[B1] Loading {N} images...")
    images = load_images(IMAGE_PATHS)
    print(f"     {N} images loaded OK.")

    print(f"\n[B2] Computing {N-1} pairwise homographies...")
    homographies = compute_pairwise_homographies(images)

    print(f"\n[B3] Computing global canvas for {N} images...")
    global_Hs, canvas_w, canvas_h, T, off_x, off_y = compute_global_canvas(images, homographies)
    print(f"     Canvas size  : {canvas_w} x {canvas_h} px")
    print(f"     Global offset: ({off_x}, {off_y}) px")
    for k, H in enumerate(global_Hs):
        print(f"     H_global[{k}] translation: ({H[0,2]:.1f}, {H[1,2]:.1f})")

    print(f"\n[B4] Warping and blending {N} images...")
    panorama_raw = warp_and_blend_all(images, global_Hs, canvas_w, canvas_h, T)

    print(f"\n[B5] Cropping bounding box...")
    panorama = crop_bounding_box(panorama_raw)
    h_out, w_out = panorama.shape[:2]
    print(f"     Raw canvas   : {canvas_w} x {canvas_h} px")
    print(f"     After crop   : {w_out} x {h_out} px")

    cv2.imwrite(OUTPUT_PATH, panorama)
    print(f"\n[B6] Done. Saved: {OUTPUT_PATH}")
    print(f"     Output resolution: {w_out} x {h_out} px")
