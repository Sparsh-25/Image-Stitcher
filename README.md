# Panoramic Image Stitcher

A classical computer vision pipeline that stitches two or more overlapping photos into a single wide panorama — built from scratch using OpenCV, with no deep learning or pretrained models.

Built as a semester project to understand the fundamentals of feature detection, geometric estimation, and image blending.

---

## How It Works

The pipeline has three sequential steps:

### Step 1 — Feature Matching (`feature_matcher1.py`)
- Detects keypoints in both images using **SIFT** (Scale-Invariant Feature Transform)
- Computes 128-dimensional descriptors for each keypoint
- Matches descriptors using **Brute-Force L2** nearest neighbour
- Filters matches using **Lowe's Ratio Test** (τ = 0.7) to discard ambiguous correspondences

### Step 2 — Homography Estimation (`homography_estimator1.py`)
- Takes the filtered point correspondences from Step 1
- Estimates a **3×3 Homography matrix** H using **MAGSAC++** (a robust RANSAC variant)
- H maps every pixel coordinate in Image A into Image B's coordinate frame
- Reports inlier ratio and mean reprojection error as quality metrics

### Step 3 — Warping & Stitching (`image_stitcher1.py`)
- Projects Image A's four corners through H to compute the exact canvas size needed (avoids clipping)
- Applies a translation offset so no pixels land at negative coordinates
- Warps Image A using `cv2.warpPerspective` with the composed transform `T @ H`
- Places Image B at its offset on the canvas
- Trims the outer black border

---

## What We Used

| Component | Tool |
|---|---|
| Language | Python 3 |
| Computer Vision | OpenCV (`cv2`) |
| Numerical Computation | NumPy |
| Visualisation | Matplotlib |
| Feature Detector | SIFT (`cv2.SIFT_create`) |
| Descriptor Matcher | BFMatcher (L2 norm) |
| Robust Estimator | MAGSAC++ (`cv2.USAC_MAGSAC`) |

No neural networks, no pretrained weights, no external datasets.

---

## How to Use

### Requirements

```bash
pip install opencv-contrib-python numpy matplotlib
```

> **Note:** `opencv-contrib-python` is required for SIFT (it's not in the base `opencv-python` package).

### Run the full Phase 1 pipeline

```bash
python image_stitcher1.py
```

The script expects two images at:
```
images/left_room.jpg
images/right_room.jpg
```

Edit the paths at the bottom of `image_stitcher1.py` to point to your own images:

```python
path_a = "/path/to/your/left_image.jpg"
path_b = "/path/to/your/right_image.jpg"
```

**Output files:**
- `final_panorama.jpg` — the stitched and cropped panorama
- `final_panorama_plot.png` — same image displayed with Matplotlib

### Run individual steps

Each module has a standalone demo you can run independently:

```bash
python feature_matcher1.py      # Visualise SIFT matches between two images
python homography_estimator1.py # Print H matrix + inlier/outlier visualisation
python image_stitcher1.py       # Full stitch
```

---

## Image Capture Guidelines

Classical stitching is very sensitive to how the photos are taken. For best results:

1. **Stand still** — don't move your feet between shots
2. **Rotate only your wrist**, not your body or shoulder
3. **Lock exposure and focus** — tap and hold on iPhone/Android before shooting
4. **Keep zoom at 1×** — same focal length for both shots
5. **Overlap by ~30–40%** — enough shared content for SIFT to find correspondences

If the camera physically moves sideways between shots (parallax), the homography model breaks down and the output will be distorted.

---

## Limitations

**1. Assumes pure camera rotation**  
The homography model (3×3 matrix, 8 degrees of freedom) is only valid when the camera rotates around its optical centre with zero translation. Physical movement between shots causes parallax, which a homography cannot represent. The output will be stretched or sheared.

**2. Sensitive to low-texture scenes**  
SIFT needs local distinctiveness — corners, blobs, edges. Smooth walls, uniform ceilings, or overcast skies produce very few keypoints and the pipeline may fail with fewer than 4 matches.

**3. No automatic exposure correction (Phase 1)**  
If your two images have different brightness or white balance, a visible seam will appear at the overlap boundary. Phase 2 (Laplacian blending + histogram exposure matching) addresses this.

**4. No support for curved distortion (Phase 1)**  
For panoramas with more than ~60° horizontal rotation, perspective distortion builds up at the edges. Cylindrical projection (Phase 2) corrects this.

**5. Manual file paths**  
Image paths are currently hardcoded in each script's `__main__` block. A CLI argument interface is on the roadmap.

---

## Project Structure

```
Image Stitcher/
├── feature_matcher1.py       — Step 1: SIFT detection + Lowe's Ratio Test
├── homography_estimator1.py  — Step 2: MAGSAC++ homography + reprojection metrics
├── image_stitcher1.py        — Step 3: Canvas computation + warpPerspective + crop
└── images/                   — Place your input images here
```

---

## Sample Output

Running on a pair of overlapping room photos:

```
Good matches         : 627
Inliers (MAGSAC++)   : 464 / 627  (74.0%)
Mean Reproj. Error   : 2.094 px
Canvas size          : 4054 × 4733 px
After border crop    : 4053 × 4733 px
```

---

*Classical Computer Vision project — 3rd Year AIML undergraduate*
