import cv2
import numpy as np
import matplotlib.pyplot as plt


class HomographyEstimator:

    def __init__(self, max_reproj_error: float = 5.0, confidence: float = 0.999):    #Initializes parameters that control how strict RANSAC (MAGSAC++) will be while estimating homography.
    
        self.max_reproj_error = max_reproj_error
        self.confidence = confidence

    def estimate(self, pts_a: np.ndarray, pts_b: np.ndarray) -> tuple:
        
        if len(pts_a) < 4:
            raise ValueError(
                f"Need at least 4 point pairs to compute a homography. "
                f"Got {len(pts_a)}. Add more images or lower Lowe's ratio threshold."
            )


        H, mask = cv2.findHomography(
            srcPoints=pts_a,
            dstPoints=pts_b,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self.max_reproj_error,   # large = outlier, small = inlier, pixels of mismatch
            confidence=self.confidence,                    # probability that the algorithm will find a solution
            maxIters=5000                                  # maximum number of iterations
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



def visualize_inliers(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    mask: np.ndarray,
    output_path: str = None
) -> None:
   
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



if __name__ == "__main__":
    import sys
    from feature_matcher1 import FeatureMatcher, visualize_matches

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
