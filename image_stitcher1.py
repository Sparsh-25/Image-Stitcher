
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageStitcher:


    def __init__(self, interpolation: int = cv2.INTER_LINEAR):
   
        self.interpolation = interpolation

    def _compute_canvas_size(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        H: np.ndarray
    ) -> tuple:
        
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]

        # The 4 corners of Image A in homogeneous coordinates (shape: 4×1×2).
        corners_a = np.float32([
            [0,      0    ],
            [w_a-1,  0    ],
            [0,      h_a-1],
            [w_a-1,  h_a-1]
        ]).reshape(-1, 1, 2)

        # Project Image A's corners through H into Image B's coordinate frame.
        # perspectiveTransform applies H and normalizes by w (homogeneous division).
        projected = cv2.perspectiveTransform(corners_a, H)  # shape: (4, 1, 2)

        # Image B's corners in its own coordinate frame (no transform needed).
        corners_b = np.float32([
            [0,      0    ],
            [w_b-1,  0    ],
            [0,      h_b-1],
            [w_b-1,  h_b-1]
        ]).reshape(-1, 1, 2)

        # Combine all corners to get the full bounding box of the panorama.
        all_corners = np.concatenate([projected, corners_b], axis=0)

        x_min, y_min = all_corners[:, 0, :].min(axis=0)
        x_max, y_max = all_corners[:, 0, :].max(axis=0)

        # If any coordinate is negative, we need to shift everything right/down
        # by (|x_min|, |y_min|) so the top-left of the canvas is (0, 0).
        offset_x = max(0, -x_min)
        offset_y = max(0, -y_min)

        canvas_w = int(np.ceil(x_max + offset_x)) + 1
        canvas_h = int(np.ceil(y_max + offset_y)) + 1

        # Translation matrix T: shifts all warped coordinates by (offset_x, offset_y).
        # We compose T @ H so that warpPerspective applies both in one pass.
        T = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1       ]
        ], dtype=np.float64)

        return canvas_w, canvas_h, T, int(offset_x), int(offset_y)

    def stitch(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        H: np.ndarray
    ) -> np.ndarray:
    

        h_b, w_b = img_b.shape[:2]
        canvas_w, canvas_h, T, off_x, off_y = self._compute_canvas_size(img_a, img_b, H)

        print(f"         Canvas size : {canvas_w} x {canvas_h} px")
        print(f"         Offset (x,y): ({off_x}, {off_y}) px  ← translation compensation")

        
        TH = T @ H

        warped_a = cv2.warpPerspective(
            src=img_a,
            M=TH,
            dsize=(canvas_w, canvas_h),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

       
        panorama = warped_a.copy()
        panorama[off_y:off_y+h_b, off_x:off_x+w_b] = img_b

       

        return panorama


    def crop_black_borders(self, panorama: np.ndarray) -> np.ndarray:
     
        # Convert to grayscale and threshold to find non-black pixels.
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

        # np.any(axis=...) checks for any non-zero value per row/column.
        non_zero_cols = np.any(gray > 0, axis=0)  # True for cols with content
        non_zero_rows = np.any(gray > 0, axis=1)  # True for rows with content

        # Find the first and last column/row that has actual content.
        col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]
        row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]

        return panorama[row_start:row_end+1, col_start:col_end+1]


# ════════════════════════════════════════════════════════════════════════════
# Visualization Utility
# ════════════════════════════════════════════════════════════════════════════

def visualize_panorama(
    panorama: np.ndarray,
    output_path: str = None
) -> None:
    """Display the stitched panorama and optionally save it."""
    h, w = panorama.shape[:2]

    plt.figure(figsize=(18, 6))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Phase 1 — Basic Panoramic Stitch\n"
        f"Output Resolution: {w} x {h} px",
        fontsize=13
    )
    plt.axis("off")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Panorama saved to: {output_path}")

    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Quick Demo  (run as: python image_stitcher.py)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from feature_matcher1 import FeatureMatcher
    from homography_estimator1 import HomographyEstimator, print_homography_report

    path_a = "/Users/Antino/Desktop/Image Stitcher/images/left_room.jpg"
    path_b = "/Users/Antino/Desktop/Image Stitcher/images/right_room.jpg"

    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    if img_a is None:
        print(f"[ERROR] Could not load Image A from: {path_a}"); sys.exit(1)
    if img_b is None:
        print(f"[ERROR] Could not load Image B from: {path_b}"); sys.exit(1)

    # ── Step 1: Feature Matching ──────────────────────────────────────────
    print("[Step 1] SIFT feature extraction and matching...")
    fm = FeatureMatcher(detector_type="SIFT", ratio_threshold=0.7)
    kps_a, desc_a = fm.detect_and_describe(img_a)
    kps_b, desc_b = fm.detect_and_describe(img_b)
    good_matches   = fm.match(desc_a, desc_b)
    pts_a, pts_b   = fm.extract_point_pairs(kps_a, kps_b, good_matches)
    print(f"         Good matches: {len(good_matches)}")

    # ── Step 2: Homography Estimation ────────────────────────────────────
    print("[Step 2] Estimating Homography with MAGSAC++...")
    estimator = HomographyEstimator(max_reproj_error=5.0, confidence=0.999)
    H, mask   = estimator.estimate(pts_a, pts_b)
    stats     = estimator.compute_reprojection_errors(pts_a, pts_b, H, mask)
    print_homography_report(H, stats)

    # ── Step 3: Warp & Stitch ─────────────────────────────────────────────
    print("[Step 3] Warping Image A and compositing onto canvas...")
    stitcher  = ImageStitcher(interpolation=cv2.INTER_LINEAR)
    panorama  = stitcher.stitch(img_a, img_b, H)

    print("[Step 3] Cropping black borders...")
    panorama_cropped = stitcher.crop_black_borders(panorama)

    h_raw, w_raw = panorama.shape[:2]
    h_out, w_out = panorama_cropped.shape[:2]
    print(f"         Raw canvas   : {w_raw} x {h_raw} px")
    print(f"         After crop   : {w_out} x {h_out} px")

    cv2.imwrite("panorama_step3.jpg", panorama_cropped)
    print("[INFO] Panorama saved to: panorama_step3.jpg")

    visualize_panorama(panorama_cropped, output_path="panorama_step3_plot.png")


