# Detailed Breakdown of Pipeline Modules

This document explicitly defines the **Output**, the **Explanation of what it means**, and **How it is used in the pipeline** for every major module we built.

---

## 1. SIFT and Feature Matcher (`feature_matcher1.py`)

*   **Output:** `pts_a` and `pts_b`. 
    These are raw NumPy arrays of shape `[N, 1, 2]`. They contain the exact `(x, y)` pixel coordinates of matching objects between Image A and Image B.
*   **What it means & What we applied:** 
    SIFT (Scale-Invariant Feature Transform) looks at an image and mathematically finds thousands of "points of interest" (like the sharp corner of a window or a high-contrast dot). It then computes a 128-number "fingerprint" (descriptor) for each point. Our `FeatureMatcher` uses a Brute-Force Matcher to compare the fingerprints from Image A against Image B. To filter out bad matches (like repeating textures on a brick wall), we applied **Lowe's Ratio Test**, which deletes any match that isn't extremely unique.
*   **How it is helpful/used:** 
    The entire geometric pipeline is useless if the computer doesn't know *where* the photos physically overlap. SIFT gives us mathematically proven coordinate pairs. For example, it tells the computer: *"Pixel (105, 400) in Image A is exactly the same physical object as Pixel (800, 421) in Image B."*

---

## 2. Homography and Homography Estimator (`homography_estimator1.py`)

*   **Output:** `H` (The 3x3 Homography Matrix) and an `inlier_mask`.
*   **What it means & What we applied:** 
    A Homography is a mathematical transformation matrix that dictates exactly how a flat plane in 3D space twists, scales, and rotates when viewed from a different camera angle. Our `HomographyEstimator` class takes the raw `(x, y)` pixel coordinates from SIFT and passes them into an algorithm called **MAGSAC++**. MAGSAC++ acts as a geometric referee—it tests the coordinates, throws away incorrect SIFT matches (outliers), and uses only the geometrically perfect matches (inliers) to solve for the final 3x3 $H$ matrix.
*   **How it is helpful/used:** 
    Without the $H$ matrix, we cannot project the images into the same perspective. `H` is the "camera rotation" formula. By mathematically multiplying the pixels of Image A by $H$, Image A physically warps and bends so that its perspective perfectly matches Image B.

---

## 3. Image Stitcher 1 (`image_stitcher1.py`)

*   **Output:** `warped_a` (the distorted image), `canvas_b` (the base image on a black array), and a translation matrix `T`.
*   **What it means & What we applied:** 
    This was our Phase 1 prototype built exclusively for stitching exactly 2 images together without any blending. First, we mathematically projected the 4 corners of Image A through the Homography matrix $H$ to see where they would land. If Image A landed in negative coordinates (e.g., coordinates off the screen to the left at $x = -1000$), OpenCV would silently clip it off and destroy half the image. To fix this, we calculated a **Translation offset (`T`)** to mathematically push the entire canvas to the right into positive space. Finally, we used `cv2.warpPerspective` to warp Image A, and used standard NumPy slicing to paste Image B directly on top.
*   **How it is helpful/used:** 
    This is the core geometrical engine. It proved we could successfully align and overlay two different photos into a massive combined canvas. Its output is perfectly geometrically aligned, but visually jarring because the seams (jump in lighting/color) are unhidden.

---

## 4. Laplacian Blender (`laplacian_blender2.py`)

*   **Output:** `panorama_blended.jpg`, a completely seamless, cross-faded BGR image.
*   **What it means & What we applied:** 
    This module was our attempt to erase the ugly visual "seam line" between two differently-lit photos. We applied three things:
    1.  **Histogram Equalization (YCrCb):** We extracted luminance (Y) and perfectly matched the brightness of Image A to Image B.
    2.  **Optimal Seam Finder:** We scanned the overlap zone to find the vertical line where the pixels were most identical.
    3.  **Laplacian Pyramids:** The algorithm mathematically split the images into 6 separate frequency layers—from broad blurry colors to ultra-sharp edges. It then blended the layers separately so that broad colors cross-faded slowly over a massive distance, while sharp edges transitioned sharply to prevent "ghosting".
*   **How it is helpful/used:** 
    It creates an invisible transition for 2 images. **However, it failed for $N$-image panoramas.** We discovered it suffered a fatal flaw when applied to massive black panoramas: floating-point numbers "bled" into the black void during the heavy Gaussian blurring iterations, completely ruining our automated boundary-cropping algorithms.

---

## 5. Multi Stitcher (`multi_stitcher2.py`)

*   **Output:** `panorama_multi_new.jpg` (The infinitely scalable, cropped, N-image panorama).
*   **What it means & What we applied:** 
    This is the grand conductor of Phase 2. It accepts an infinite array of images (`left`, `middle`, `right`...) and manages the entire pipeline.
    We applied two major breakthroughs here:
    1.  **Center-Anchoring:** If you anchor 5 images linearly onto Image 0, the scale factor multiplies iteratively until Image 5 is 20x larger than it should be. Center-Anchoring sets the *middle* image as the Identity matrix, pulling the left images forward and sending right images backward via Inverse Homographies. This beautifully halved the cumulative scale distortion.
    2.  **Distance-Transform Weighted Accumulator:** We threw away the failed Laplacian Blender entirely. Instead, we used a Distance Transform to generate an alpha mask based purely on *how close a pixel is to the center of its original photo*. We dumped every warped image into a massive `float32` accumulator canvas.
*   **How it is helpful/used:** 
    Because a pixel in the center of a glass lens is optically superior to the distorted edges, the Distance-Transform mathematically guarantees that the computer trusts center pixels more. This natively creates an infinitely wide, mathematically perfect 50/50 cross-fade right down the center of any overlap zone, ignoring the Laplacian "black-canvas float bug" entirely.
