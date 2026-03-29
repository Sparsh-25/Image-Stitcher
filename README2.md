# Technical Deep Dive — Panoramic Image Stitcher

This document explains every technique used in the pipeline: what it is, the math behind it, why we chose it, and what breaks without it.

---

## Table of Contents

1. [SIFT Feature Detection](#1-sift-feature-detection)
2. [Lowe's Ratio Test](#2-lowes-ratio-test)
3. [Homography Matrix — What It Is](#3-homography-matrix)
4. [RANSAC and MAGSAC++](#4-ransac-and-magsac)
5. [Corner-Projection Canvas](#5-corner-projection-canvas)
6. [Exposure Matching](#6-exposure-matching)
7. [Gradient Blend Mask](#7-gradient-blend-mask)
8. [Laplacian Pyramid Blending](#8-laplacian-pyramid-blending)
9. [N-Image Homography Chaining](#9-n-image-homography-chaining)
10. [Bounding Box Crop](#10-bounding-box-crop)
11. [Why Not Deep Learning](#11-why-not-deep-learning)
12. [Known Limitations](#12-known-limitations)

---

## 1. SIFT Feature Detection

**File:** `feature_matcher1.py`

### What it does
SIFT (Scale-Invariant Feature Transform, Lowe 2004) detects *keypoints* — locally distinctive positions in an image like corners, blobs, and edges — and computes a **128-dimensional descriptor** for each one that encodes the local gradient pattern.

### Math

**Scale-space construction:**  
The image is convolved with Gaussians at increasing sigma values across multiple octaves (halved resolutions). Subtracting adjacent Gaussian-blurred images gives the Difference of Gaussians (DoG):

```
DoG(x, y, σ) = G(x, y, kσ) * I(x,y)  −  G(x, y, σ) * I(x,y)
```

Keypoints are detected as local extrema (maxima or minima) in this DoG scale-space — points that are brighter or darker than all 26 neighbours (8 in-plane + 9 above + 9 below).

**Orientation assignment:**  
A dominant gradient direction is computed from the local neighbourhood and assigned to the keypoint. This makes the descriptor rotation-invariant — the 128D vector is always computed relative to this dominant angle.

**Scale invariance:**  
Because the extremum is found across scale levels, the same physical blob is detected at the same DoG level regardless of zoom or distance.

### Parameters we tuned

| Parameter | Value | Why |
|---|---|---|
| `nfeatures` | 8000 | Cap at strongest keypoints. Unlimited → 50k+ on 12MP, many noisy clusters |
| `contrastThreshold` | 0.04 | Filters weak-contrast keypoints. Lowered to 0.02 for poor-quality images |
| `nOctaveLayers` | 3 | Number of scale levels per octave |
| `sigma` | 1.6 | Lowe's recommended initial blur |

### Why not ORB?
ORB (Oriented FAST and Rotated BRIEF) is ~5× faster, uses binary descriptors, and has no patent restrictions. We tested it — 82% inlier ratio vs SIFT's 74%, but ORB's reprojection error was measurably higher. For blending quality, pixel-precise correspondences matter more than speed.

---

## 2. Lowe's Ratio Test

**File:** `feature_matcher1.py`

### What it does
Filters ambiguous matches. For each keypoint in Image A, we find its **two** nearest neighbours in Image B and apply the ratio test:

```
accept match  if  d(best) / d(second_best)  <  τ
```

### Why it works
If a keypoint's best match is much closer than the second-best (ratio << 1), the match is *distinctly* better — reliable. If the two nearest matches are almost equidistant (ratio ≈ 1), the keypoint lies near a repetitive texture (like a tiled wall or striped fabric) where multiple regions look equally similar → ambiguous → discard.

### Why not crossCheck?
`BFMatcher(crossCheck=True)` returns a match `(A→B)` only if B also picks A back as its nearest neighbour. This is more conservative but gives far fewer matches — in our first test, 12 matches vs 627 with the ratio test.

### Threshold tuning

| τ | Matches | Inlier ratio |
|---|---|---|
| 0.60 | 312 | 82% |
| 0.70 | 627 | 74% |
| 0.75 | 880 | 66% |
| 0.80 | 1200 | 58% |

We used **τ = 0.7** — Lowe's own recommendation.

---

## 3. Homography Matrix

**File:** `homography_estimator1.py`

### What it is
A **3×3 matrix H** that maps any point in Image A's pixel coordinate frame to the corresponding point in Image B's frame. It encodes rotation, scale, shear, and perspective projection in one transform.

```
[x']     [h00  h01  h02]   [x]
[y']  =  [h10  h11  h12] * [y]
[w']     [h20  h21  1  ]   [1]

Result:  (x'/w',  y'/w')
```

The division by `w'` (homogeneous normalisation) is what allows perspective (non-affine) warping.

### Degrees of freedom
H is a 3×3 matrix with 9 entries. The bottom-right element is fixed to 1 (scale normalisation), leaving **8 degrees of freedom**. Each point correspondence gives 2 equations, so you need a minimum of **4 point pairs** to solve for H.

### Direct Linear Transform (DLT)
Our first implementation. Each point pair `(x, y) → (x', y')` gives two constraint equations:

```
x' (h20·x + h21·y + h22) = h00·x + h01·y + h02
y' (h20·x + h21·y + h22) = h10·x + h11·y + h12
```

Stack N such equations into matrix A (2N × 9), then solve `A·h = 0` via SVD. The eigenvector corresponding to the smallest singular value is H.

**Problem:** DLT treats all correspondences equally. One bad match pulls the entire solution. With real noisy data, DLT fails.

---

## 4. RANSAC and MAGSAC++

**File:** `homography_estimator1.py`

### RANSAC
Random Sample Consensus. Given a set of noisy matches, it:

1. Randomly picks **4** point pairs (the minimum to fit H)
2. Fits H using DLT on those 4 points
3. Tests all remaining N points: a point is an **inlier** if its reprojection error is below threshold `ε`
4. Records `(H, inlier_count)`
5. Repeats for `maxIters` iterations
6. Returns H with the **most inliers**

**Reprojection error** for a match `(pA, pB)`:
```
error = || H * pA  −  pB ||₂   (L2 pixel distance)
```

Number of iterations needed for 99% confidence:
```
k = log(1 - p) / log(1 - (1-e)^4)
```
where `p = confidence` (0.999), `e = outlier_ratio`.

### MAGSAC++
A strict improvement over RANSAC. Where RANSAC uses a hard binary inlier/outlier threshold, MAGSAC++ scores each point with a **continuous weight** based on its error relative to a noise model:

```
weight(i) = 1 - F(error_i / σ)
```

where `F` is the CDF of the noise distribution. Points far from the fitted H contribute near-zero weight; points close contribute near-1 weight. This gives a globally consistent solution rather than one sensitive to the exact random seeds RANSAC draws.

**Result on our images:** RANSAC sometimes gave 55% inlier ratio on bad random draws. MAGSAC++ consistently gave 74%, same answer every run.

### Our parameters

```python
cv2.findHomography(
    pts_a, pts_b,
    method=cv2.USAC_MAGSAC,
    ransacReprojThreshold=5.0,   # ε: inlier threshold in pixels
    confidence=0.999,            # p: probability of finding true H
    maxIters=5000
)
```

| `ransacReprojThreshold` | Inlier ratio | Mean error |
|---|---|---|
| 2.0 px | 52% | 0.6 px |
| 5.0 px | 74% | 2.1 px |
| 8.0 px | 81% | 2.4 px |

At 2px, correct matches with JPEG compression noise are rejected as outliers. At 8px, wrong matches pollute H. **5px** is optimal for mobile-camera JPEG images.

---

## 5. Corner-Projection Canvas

**File:** `image_stitcher1.py`

### The problem
`cv2.warpPerspective` places the output at the coordinates H maps to. If H has a negative x-translation (Image A is geometrically to the right of Image B), Image A lands at negative pixel coordinates — which OpenCV silently clips to black.

Naive canvas:
```python
canvas_w = img_a.width + img_b.width   # Wrong for negative translations
```

### The fix
Before allocating the canvas, project Image A's 4 corners through H:

```python
corners_a = [[0,0], [w,0], [0,h], [w,h]]   # 4 corners of Image A
projected = cv2.perspectiveTransform(corners_a, H)  # Where they land in B's frame
```

Compute the bounding box over all projected corners + Image B's corners. If `x_min < 0`, we need to shift everything right by `|x_min|`. Build a translation matrix T:

```
T = [[1,  0,  offset_x],
     [0,  1,  offset_y],
     [0,  0,  1       ]]
```

Apply: `warpPerspective(img_a, T @ H, canvas_size)`

This composites both transforms in a single warp call — no double-interpolation.

---

## 6. Exposure Matching

**File:** `laplacian_blender2.py` → `match_exposure()`

### The problem
Two photos of the same scene taken 2 seconds apart can have different brightness because phone cameras auto-adjust exposure. Even with geometrically perfect alignment, different luminance values create a visible seam.

### Histogram Matching (CDF Method)

We work in **YCrCb** colorspace — Y is luminance, Cr and Cb are chrominance. We only modify Y, leaving colour unchanged.

**Step 1:** Compute the normalised histogram (probability mass function) of the Y channel for src (Image A) and ref (Image B):

```
hist_src[i] = count of pixels with Y == i  /  total_pixels
```

**Step 2:** Compute cumulative distribution functions:
```
CDF_src[i] = Σ hist_src[j]  for j = 0..i
CDF_ref[i] = Σ hist_ref[j]  for j = 0..i
```

**Step 3:** Build a 256-entry LUT: for each intensity `i` in src, find the intensity `j` in ref whose CDF value is closest:
```
LUT[i] = min_j { |CDF_ref[j] - CDF_src[i]| }
```
This is the **histogram specification** transform — it remaps src's luminance distribution to match ref's.

**Step 4:** Apply LUT to src's Y channel → convert back to BGR.

**Result on our images:** Luminance gap in the overlap zone reduced from 7.3 → 0.3 (96% reduction).

---

## 7. Gradient Blend Mask

**File:** `laplacian_blender2.py` → `build_blend_mask()`

### The problem
After warping, two images overlap in a shared zone. A hard binary mask:
```
mask = 0  in Image A zone
mask = 1  in Image B zone
```
leaves a sharp edge at the seam that the Laplacian pyramid cannot smooth at its finest level.

### The fix
Build a linear gradient across a narrow seam band instead of the full overlap:

```python
seam_mid  = (col_left + col_right) // 2   # midpoint of overlap
band_left = seam_mid - 25                  # 50px wide band
band_right= seam_mid + 25

gradient[col] = clip((col - band_left) / 50, 0.0, 1.0)
```

`mask[row, col] = gradient[col] * has_b[row, col]`

The narrower the band, the less ghosting — but also the sharper the visible cut if the images don't align perfectly. **50px** is our working value for 12MP room images.

### Why NumPy, not a Python loop?
```python
# Python loop (old)  — 6000 cols × Python overhead = ~5 seconds
for col in range(W):
    gradient[col] = ...

# NumPy (new) — vectorised C operation = ~8 milliseconds
col_idx  = np.arange(W, dtype=np.float32)
gradient = np.clip((col_idx - band_left) / band_w, 0.0, 1.0)
```
~625× speedup.

---

## 8. Laplacian Pyramid Blending

**File:** `laplacian_blender2.py` → `LaplacianBlender`

### Core idea
A single pixel-level blend (even with a gradient) fails because natural images contain structure at multiple spatial scales. The Laplacian pyramid decomposes an image into frequency bands — each band is blended separately at the right spatial scale.

### Gaussian Pyramid
Repeatedly downsample by 2× using `cv2.pyrDown` (Gaussian blur then subsample):

```
G0 = original image        (full resolution)
G1 = pyrDown(G0)           (half resolution)
G2 = pyrDown(G1)           (quarter resolution)
...
GN = coarsest level
```

### Laplacian Pyramid
Each level captures the fine detail *lost* in the downsampling step:

```
L0 = G0 - pyrUp(G1)        (finest detail: edges, texture)
L1 = G1 - pyrUp(G2)        (medium structures)
...
LN = GN                    (coarsest: global colour/brightness)
```

`pyrUp` doubles resolution, but is **not** a perfect inverse of `pyrDown` for odd-sized dimensions. We always `cv2.resize` after `pyrUp` to match the exact dimensions of the finer level.

### Per-level Blending
Build a Gaussian pyramid of the blend mask M. Blend each Laplacian level:

```
Blended[k] = M[k] * L_B[k]  +  (1 - M[k]) * L_A[k]
```

At coarse level `LN`, `M[N]` is heavily blurred → wide, smooth colour transition.  
At fine level `L0`, `M[0]` is a tight gradient → narrow detail transition.

### Pyramid Collapse
Reconstruct full resolution from the blended pyramid:

```
result = blended[N]
for k in N-1..0:
    result = pyrUp(result)
    result = resize(result, shape_of_L[k])
    result = result + blended[k]

output = clip(result, 0, 255).astype(uint8)
```

### Why 6 levels?
| Levels | Seam | Speed |
|---|---|---|
| 3 | Visible at fine details | Fast |
| 6 | Invisible on well-captured images | Moderate |
| 9 | Marginally better | Slow |

6 levels gives a coarsest level of `original / 2^6 = ~47px` for a 3024px image — enough to smooth global brightness differences.

---

## 9. N-Image Homography Chaining

**File:** `multi_stitcher2.py`

### The problem
For N images we compute N−1 pairwise homographies. But each H maps only `img_k → img_{k+1}`. To warp all images onto a single canvas, we need a *global* transform for each image relative to a fixed anchor (image 0).

### The chain
```
H_global[0] = Identity           (image 0 is the anchor)
H_global[1] = H[0]               (img 1 → img 0 frame)
H_global[2] = H_global[1] @ H[0→1]  =  H[0] @ H[1]
H_global[k] = H[0] @ H[1] @ ... @ H[k-1]
```

Matrix multiplication is not commutative (`A@B ≠ B@A`), so order matters. `H[k-1]` is applied first (innermost), taking image k to image k−1's frame, then `H_global[k-1]` takes it to image 0's frame.

### Global canvas
Project all 4 corners of all N images through their `H_global`:
```
for each image k:
    corners_k → perspectiveTransform(corners_k, H_global[k])

all_corners = concatenate all projected corners
x_min, y_min = min over all_corners
x_max, y_max = max over all_corners

canvas_w = ceil(x_max - x_min) + 1
canvas_h = ceil(y_max - y_min) + 1
T = translation by (-x_min, -y_min)
```

### Incremental blending
Images are blended one at a time into an accumulator canvas:
```
accumulator = warp(img_0, T @ H_global[0])
for k = 1..N-1:
    warped_k = warp(exposure_matched(img_k), T @ H_global[k])
    mask = build_blend_mask(accumulator, warped_k)
    accumulator = LaplacianBlend(accumulator, warped_k, mask)
```

---

## 10. Bounding Box Crop

**File:** `multi_stitcher2.py` → `crop_bounding_box()`

After warping, the canvas has black padding everywhere no image reached. Crop to the bounding box of non-zero pixels:

```python
gray = cvtColor(panorama, BGR2GRAY)
rows = any(gray > 0, axis=1)   # rows containing at least 1 non-black pixel
cols = any(gray > 0, axis=0)   # cols containing at least 1 non-black pixel

r0, r1 = first and last True row
c0, c1 = first and last True col

cropped = panorama[r0:r1, c0:c1]
```

This is O(H×W), runs in milliseconds, and retains the maximum possible image area. May retain small black corner triangles from the warp boundary — acceptable for a final panorama.

---

## 11. Why Not Deep Learning

We deliberately avoided neural networks for three reasons:

1. **Interpretability** — every step in this pipeline has a closed-form mathematical explanation. When something fails, we can print the homography matrix, inspect match quality, and trace the exact cause. Neural network failures have no such leverage.

2. **Data and compute** — deep stitching models require millions of paired training samples and GPU training. We had a laptop and 3 room photos.

3. **Understanding** — fine-tuning pretrained models was already most of our coursework. Building from linear algebra and gradient descent upward was the actual learning objective.

---

## 12. Known Limitations

### Parallax (not fixable by blending)
A 3×3 homography assumes the camera rotates around a fixed optical centre with **zero translation**. When the camera physically moves (even slightly), different-depth objects shift by different amounts — parallax. The warped image looks "folded" and objects near the seam appear doubled. No amount of blending corrects this: it is a geometric model mismatch.

**Fix:** Shoot images by rotating the camera in place — wrist rotation only, feet planted.

### Wide overlap → ghosting
The blend mask gradient is only effective when it spans a narrow region. If two images overlap 60%+ of the frame, even a 50px gradient leaves visible seam transitions because the homography can't perfectly align parallax-affected points near the seam.

### Auto-exposure differences
Histogram matching corrects global luminance shifts. It doesn't handle local exposure variation (e.g., a window in one image and a wall in the other — the luminance distribution has different *shape*, not just a different offset). The LUT remaps globally, so bright window regions may over-expose after matching.

### `nfeatures=8000` on low-texture scenes
If the overlap zone happens to contain a smooth white wall or uniform ceiling, SIFT will find very few keypoints there specifically, even with 8000 total keypoints distributed across the image. The pipeline may fail with < 4 matches. Adaptive keypoint density per image region would help but is not implemented.

---

*All code is in pure Python + OpenCV + NumPy. No pretrained weights, no GPU required.*
