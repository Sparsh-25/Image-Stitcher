"""
homography_estimator.py
=======================
Phase 1 – Step 2: Homography Estimation with MAGSAC++

Architectural Choice: MAGSAC++ over standard RANSAC
- Standard RANSAC uses a hard pixel threshold to declare inliers/outliers.
  This makes it brittle: a 1-pixel change in the threshold can swap a point
  between "inlier" and "outlier", destabilizing the final homography.
- MAGSAC++ replaces the hard threshold with a soft probabilistic weighting.
  Each correspondence is weighted by its probability of being an inlier.
  The homography is then fitted by WEIGHTED least squares, extracting maximum
  information from every correspondence — critical when we have few matches.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class HomographyEstimator:
    """
    Estimates the 3×3 projective homography matrix H that maps points
    from Image A into the coordinate frame of Image B, using MAGSAC++
    as the robust outlier-rejection method.

    Pipeline:
        pts_a, pts_b  (from FeatureMatcher)
             │
             ▼
        MAGSAC++ findHomography
             │
             ├──► H  (3×3 homography matrix)
             └──► mask  (inlier/outlier label per correspondence)
    """

    def __init__(self, max_reproj_error: float = 5.0, confidence: float = 0.999):
        """
        Args:
            max_reproj_error: σ* — the UPPER BOUND on inlier noise (pixels).
                              In MAGSAC++, this is NOT a hard threshold but an
                              upper integration limit for the weight integral.
                              5.0 pixels is the standard recommendation.
            confidence:       Probability that the final H contains no outliers.
                              0.999 means MAGSAC++ keeps sampling until it is
                              99.9% confident the best model found is correct.
        """
        self.max_reproj_error = max_reproj_error
        self.confidence = confidence

    def estimate(self, pts_a: np.ndarray, pts_b: np.ndarray) -> tuple:
        """
        Compute the homography H such that:  pts_b ≈ H · pts_a

        Internally this calls cv2.findHomography with method=cv2.USAC_MAGSAC,
        which implements the full MAGSAC++ pipeline:
          1. Randomly sample 4 correspondences (minimum for DLT).
          2. Solve the 3×3 H via the Direct Linear Transform (DLT).
          3. Compute residuals (reprojection errors) for all correspondences.
          4. Assign soft inlier weights via the MAGSAC++ probability model.
          5. Re-estimate H via weighted least squares over ALL correspondences.
          6. Repeat until confidence threshold is met.

        Args:
            pts_a: np.ndarray shape (N, 1, 2) — keypoint coords in Image A.
            pts_b: np.ndarray shape (N, 1, 2) — corresponding coords in Image B.

        Returns:
            H:    np.ndarray shape (3, 3) — the estimated homography matrix.
            mask: np.ndarray shape (N, 1) — 1 for inliers, 0 for outliers.

        Raises:
            ValueError: If fewer than 4 correspondences are provided
                        (minimum needed to solve for 8 DoF of H).
            RuntimeError: If MAGSAC++ fails to find a valid homography.
        """
        if len(pts_a) < 4:
            raise ValueError(
                f"Need at least 4 point pairs to compute a homography. "
                f"Got {len(pts_a)}. Add more images or lower Lowe's ratio threshold."
            )

        # cv2.USAC_MAGSAC is OpenCV's implementation of MAGSAC++.
        # Parameters:
        #   srcPoints     = pts_a (source: Image A coordinates)
        #   dstPoints     = pts_b (destination: Image B coordinates)
        #   method        = cv2.USAC_MAGSAC  → use MAGSAC++ robust estimator
        #   ransacReprojThreshold = σ* (upper bound on inlier noise, in pixels)
        #   confidence    = stopping criterion probability
        #   maxIters      = 5000 iterations max (MAGSAC++ converges fast in practice)
        H, mask = cv2.findHomography(
            srcPoints=pts_a,
            dstPoints=pts_b,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self.max_reproj_error,
            confidence=self.confidence,
            maxIters=5000
        )

        if H is None:
            raise RuntimeError(
                "MAGSAC++ failed to find a valid homography. "
                "This usually means too few matches or the images don't overlap."
            )

        return H, mask

    def get_inlier_points(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        mask: np.ndarray
    ) -> tuple:
        """
        Filter point pairs to only those classified as inliers by MAGSAC++.

        Args:
            pts_a, pts_b: Full point arrays (N, 1, 2).
            mask:         Inlier mask from estimate(), shape (N, 1).

        Returns:
            inliers_a, inliers_b: Filtered arrays containing only inlier points.
        """
        # mask.ravel() converts (N,1) → (N,) for boolean indexing.
        inlier_mask = mask.ravel().astype(bool)
        return pts_a[inlier_mask], pts_b[inlier_mask]

    def compute_reprojection_errors(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        H: np.ndarray,
        mask: np.ndarray
    ) -> dict:
        """
        Compute reprojection errors for inlier correspondences to validate H.

        The reprojection error for a correspondence (p_a, p_b) is:
            e_i = || p_b  -  (H · p_a)_normalized ||₂

        where (H · p_a)_normalized divides by the homogeneous coordinate w
        to convert from projective to Euclidean space.

        Args:
            pts_a, pts_b: Full point arrays (N, 1, 2).
            H:            Estimated 3×3 homography.
            mask:         Inlier mask from estimate().

        Returns:
            dict with keys:
                'mean_error'   → mean reprojection error over inliers (pixels)
                'max_error'    → worst-case error (pixels)
                'inlier_count' → number of inliers
                'total_count'  → total correspondences
                'inlier_ratio' → inlier_count / total_count
        """
        inliers_a, inliers_b = self.get_inlier_points(pts_a, pts_b, mask)

        # cv2.perspectiveTransform applies H to each point in inliers_a.
        # It handles the homogeneous division internally:
        #   [x', y', w']ᵀ = H · [x, y, 1]ᵀ
        #   result = (x'/w', y'/w')
        projected = cv2.perspectiveTransform(inliers_a, H)  # shape (N, 1, 2)

        # L2 distance between projected point and actual point in Image B.
        errors = np.linalg.norm(inliers_b - projected, axis=2).ravel()

        return {
            "mean_error":   float(np.mean(errors)),
            "max_error":    float(np.max(errors)),
            "inlier_count": int(mask.sum()),
            "total_count":  len(pts_a),
            "inlier_ratio": float(mask.sum()) / len(pts_a)
        }


# ════════════════════════════════════════════════════════════════════════════
# Visualization Utilities
# ════════════════════════════════════════════════════════════════════════════

def visualize_inliers(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    mask: np.ndarray,
    output_path: str = None
) -> None:
    """
    Draw inlier (green) and outlier (red) correspondences side-by-side.

    Args:
        image_a, image_b: Source images (BGR).
        pts_a, pts_b:     Full point arrays (N, 1, 2).
        mask:             Inlier mask (N, 1) from MAGSAC++.
        output_path:      Optional path to save the figure.
    """
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    # Create a side-by-side canvas.
    canvas_h = max(h_a, h_b)
    canvas   = np.zeros((canvas_h, w_a + w_b, 3), dtype=np.uint8)
    canvas[:h_a, :w_a]      = image_a
    canvas[:h_b, w_a:w_a+w_b] = image_b

    inlier_mask = mask.ravel().astype(bool)

    for i, (pa, pb) in enumerate(zip(pts_a.reshape(-1, 2), pts_b.reshape(-1, 2))):
        pt_a = (int(pa[0]),        int(pa[1]))
        pt_b = (int(pb[0]) + w_a, int(pb[1]))  # offset B by width of A

        color  = (0, 220, 0) if inlier_mask[i] else (0, 0, 220)  # Green / Red
        radius = 4            if inlier_mask[i] else 3

        cv2.circle(canvas, pt_a, radius, color, -1)
        cv2.circle(canvas, pt_b, radius, color, -1)
        cv2.line(canvas, pt_a, pt_b, color, 1)

    plt.figure(figsize=(20, 8))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    inlier_count  = int(inlier_mask.sum())
    outlier_count = len(inlier_mask) - inlier_count
    plt.title(
        f"MAGSAC++ Inlier/Outlier Classification\n"
        f"Inliers (green): {inlier_count}   Outliers (red): {outlier_count}",
        fontsize=13
    )
    plt.axis("off")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Inlier visualization saved to: {output_path}")
    plt.show()


def print_homography_report(H: np.ndarray, stats: dict) -> None:
    """Pretty-print the estimated homography matrix and validation statistics."""
    print("\n" + "═" * 55)
    print("  HOMOGRAPHY MATRIX  H  (maps Image A → Image B)")
    print("═" * 55)
    for row in H:
        print(f"  [ {row[0]:+10.5f}  {row[1]:+10.5f}  {row[2]:+10.5f} ]")
    print("═" * 55)
    print(f"  Inliers          : {stats['inlier_count']} / {stats['total_count']}"
          f"  ({stats['inlier_ratio']*100:.1f}%)")
    print(f"  Mean Reproj. Err : {stats['mean_error']:.3f} px")
    print(f"  Max  Reproj. Err : {stats['max_error']:.3f} px")
    print("═" * 55 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Quick Demo  (run as: python homography_estimator.py)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from feature_matcher import FeatureMatcher, visualize_matches

    path_a = "/Users/Antino/Desktop/Image Stitcher/images/left_room.jpg"
    path_b = "/Users/Antino/Desktop/Image Stitcher/images/right_room.jpg"

    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    if img_a is None:
        print(f"[ERROR] Could not load Image A from: {path_a}"); sys.exit(1)
    if img_b is None:
        print(f"[ERROR] Could not load Image B from: {path_b}"); sys.exit(1)

    # ── Step 1: Feature Matching (reuse from previous module) ─────────────
    print("[Step 1] Running SIFT feature extraction and matching...")
    fm = FeatureMatcher(detector_type="SIFT", ratio_threshold=0.7)
    kps_a, desc_a = fm.detect_and_describe(img_a)
    kps_b, desc_b = fm.detect_and_describe(img_b)
    good_matches   = fm.match(desc_a, desc_b)
    pts_a, pts_b   = fm.extract_point_pairs(kps_a, kps_b, good_matches)
    print(f"         Good matches: {len(good_matches)}")

    # ── Step 2: Homography Estimation with MAGSAC++ ───────────────────────
    print("[Step 2] Estimating Homography with MAGSAC++...")
    estimator = HomographyEstimator(max_reproj_error=5.0, confidence=0.999)

    H, mask = estimator.estimate(pts_a, pts_b)

    stats = estimator.compute_reprojection_errors(pts_a, pts_b, H, mask)
    print_homography_report(H, stats)

    # ── Visualize inlier/outlier classification ───────────────────────────
    print("[Step 2] Visualizing MAGSAC++ inlier/outlier classification...")
    visualize_inliers(img_a, img_b, pts_a, pts_b, mask,
                      output_path="inlier_result.png")

    print("[✓] Step 2 Complete — H matrix ready for Step 3 (Warping & Stitching).")
