# Panoramic Image Stitcher

A classical computer vision pipeline that stitches two or more overlapping photos into a single wide panorama — built from scratch using OpenCV, with no deep learning or pretrained models.

Built as a semester project to understand the fundamentals of feature detection, geometric estimation, and image blending.

---

## How It Works

The pipeline is split into two phases.

---

### Phase 1 — Two-Image Stitching

#### Step 1 — Feature Matching (`feature_matcher1.py`)
- Detects keypoints in both images using **SIFT** (Scale-Invariant Feature Transform)
- Computes 128-dimensional descriptors for each keypoint
- Matches descriptors using **Brute-Force L2** nearest neighbour
- Filters matches using **Lowe's Ratio Test** (τ = 0.7) to discard ambiguous correspondences

#### Step 2 — Homography Estimation (`homography_estimator1.py`)
- Takes the filtered point correspondences from Step 1
- Estimates a **3×3 Homography matrix** H using **MAGSAC++** (a robust RANSAC variant)
- H maps every pixel coordinate in Image A into Image B's coordinate frame
- Reports inlier ratio and mean reprojection error as quality metrics

#### Step 3 — Warping & Stitching (`image_stitcher1.py`)
- Projects Image A's four corners through H to compute the exact canvas size (avoids clipping)
- Applies a translation offset so no pixels land at negative coordinates
- Warps Image A using `cv2.warpPerspective` with the composed transform `T @ H`
- Places Image B at its offset on the canvas and trims the outer black border

---

### Phase 2 — Seamless Blending & N-Image Stitching

#### Exposure Matching (`laplacian_blender2.py` → `match_exposure`)
- Converts Image A to **YCrCb** colorspace to isolate luminance from colour
- Computes the **Cumulative Distribution Function (CDF)** of the Y channel for both images
- Builds a 256-entry **histogram matching LUT** to remap Image A's brightness to match Image B
- Eliminates colour jump at the seam without shifting hues

#### Gradient Blend Mask (`laplacian_blender2.py` → `build_blend_mask`)
- Finds the **optimal seam column** in the overlap zone using vectorised per-column mean absolute difference (minimum-difference column = least visible seam)
- Builds a directional linear gradient centred at the optimal seam
- Direction (0→1 or 1→0) is auto-detected from each image's centre-of-mass column — handles both left-of-anchor and right-of-anchor images correctly

#### Laplacian Pyramid Blending (`laplacian_blender2.py` → `LaplacianBlender`)
- Decomposes both images into frequency bands: **Gaussian → Laplacian pyramid** (6 levels)
- Blends each band separately using a progressively blurred mask
  - Coarse levels (global colour) → wide smooth transition
  - Fine levels (edges, texture) → tight transition
- Collapses the blended pyramid back to full resolution
- Includes a `pyrUp` resize fix for odd-dimension images to avoid shape mismatches

#### N-Image Sequential Stitching (`multi_stitcher2.py`)

**Pairwise homographies:**  
For N images, computes N-1 pairwise H matrices (SIFT + MAGSAC++ per pair).

**Centre-anchored global chain:**  
Anchors at the **middle image** to distribute scale distortion symmetrically. Global transforms are:
```
H_global[anchor] = Identity

# RIGHT of anchor  (inv chain — maps image k back to anchor frame)
H_global[k] = H_global[k-1] @ inv(H[k-1])

# LEFT of anchor  (forward chain — maps image k forward to anchor frame)
H_global[k] = H_global[k+1] @ H[k]
```

**Distance-transform weighted accumulation:**  
All N images are blended in a single float32 weighted sum pass:
```
canvas_f    += warped_k * weight_k
weight_sum  += weight_k
result       = canvas_f / weight_sum
```
Weight per pixel = distance from the nearest image boundary (via `cv2.distanceTransform`). Pixels far from their image edge get high weight; edge pixels get near-zero weight. This gives automatic smooth transitions at all overlaps with no explicit seam code and no order dependency.

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
| Blend Mask | Distance Transform (`cv2.distanceTransform`) |

No neural networks, no pretrained weights, no external datasets.

---

## How to Use

### Requirements

```bash
pip install opencv-contrib-python numpy matplotlib
```

> **Note:** `opencv-contrib-python` is required for SIFT (it's not in the base `opencv-python` package).

### Phase 1 — Stitch two images

```bash
python image_stitcher1.py
```

Edit paths at the bottom of `image_stitcher1.py`:
```python
path_a = "/path/to/left_image.jpg"
path_b = "/path/to/right_image.jpg"
```

Output: `final_panorama.jpg`

### Phase 2 — Stitch N images (seamless)

```bash
python multi_stitcher2.py
```

Edit `IMAGE_PATHS` at the bottom of `multi_stitcher2.py`:
```python
IMAGE_PATHS = [
    "/path/to/image_left.jpg",
    "/path/to/image_centre.jpg",
    "/path/to/image_right.jpg",
]
OUTPUT_PATH = "/path/to/output.jpg"
```

List images in **left → right order**. Output: `panorama_multi.jpg`

### Run individual steps

```bash
python feature_matcher1.py       # Visualise SIFT matches
python homography_estimator1.py  # Print H matrix + reprojection metrics
python laplacian_blender2.py     # Test exposure matching + blending on 2 images
python multi_stitcher2.py        # Full N-image stitch
```

---

## Image Capture Guidelines

Classical stitching is very sensitive to how the photos are taken. For best results:

1. **Stand still** — don't move your feet between shots
2. **Rotate only your wrist**, not your body or shoulder
3. **Lock exposure and focus** — tap and hold on iPhone/Android before each shot
4. **Keep zoom at 1×** — same focal length for all shots
5. **Overlap by ~30–40%** — enough shared content for SIFT to find correspondences

If the camera physically moves sideways between shots (parallax), the homography model breaks down and the output will show doubled objects and seam distortion.

---

## Limitations

**1. Assumes pure camera rotation**  
The homography model (3×3 matrix, 8 DoF) is only valid when the camera rotates around its optical centre with zero translation. Physical movement between shots causes parallax which no blending can fully hide.

**2. Sensitive to low-texture scenes**  
SIFT needs local distinctiveness — corners, blobs, edges. Smooth walls or overcast skies produce very few keypoints and the pipeline may fail with fewer than 4 matches.

**3. No cylindrical/spherical projection**  
For panoramas wider than ~60°, perspective distortion builds up at the edges. Cylindrical pre-warping (not yet implemented) would correct this.

**4. Manual file paths**  
Image paths are hardcoded in each script's `__main__` block.

---

## Project Structure

```
Image Stitcher/
├── feature_matcher1.py       — SIFT detection + Lowe's Ratio Test
├── homography_estimator1.py  — MAGSAC++ homography + reprojection metrics
├── image_stitcher1.py        — Two-image canvas computation + warpPerspective + crop
├── laplacian_blender2.py     — Exposure matching + Laplacian pyramid blending
├── multi_stitcher2.py        — N-image pipeline: pairwise H chain + weighted blend
└── images/                   — Place your input images here
```

---

## Sample Output

Running the Phase 2 pipeline on 3 overlapping room photos:

```
Loaded 3 images (3024 × 4032 px each)

Pair (0,1): 1539 / 1885 inliers (81.6%)  MRE: 1.632 px
Pair (1,2): 1759 / 2310 inliers (76.1%)  MRE: 1.607 px

Canvas size  : 5541 × 5163 px
H_global[0]  : (-1126.3, -285.0)  ← left of anchor
H_global[1]  : (0.0, 0.0)         ← anchor (middle image)
H_global[2]  : (+852.9, +81.2)    ← right of anchor
```

---

*Classical Computer Vision project — 3rd Year AIML undergraduate*
