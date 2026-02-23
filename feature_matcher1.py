from PIL.ImageMath import imagemath_equal
import cv2 
import numpy as np
import matplotlib.pyplot as plt


class FeatureMatcher:

    def __init__(self, detector_type: str = "SIFT", ratio_threshold: float = 0.7):
        self.detector_type = detector_type.upper()
        self.ratio_threshold = ratio_threshold

        if self.detector_type == "SIFT":
            self.detector = cv2.SIFT_create(
                nfeatures=8000,
                n0ctaveLayers = 3,
                contrastThreshold = 0.04,
                edgeThreshold = 10,
                sigma = 1.6
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError(f"Usupported detector_type '{detector_type}")
    


    def detect_and_match(self, image: np.ndarray) -> tuple:

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, mask=None)

        return keypoints, descriptors

    


    def match(self, descriptors_a: np.ndarray, descriptors_b: np.ndarray) -> list:

        if descriptors_a is None or descriptors_b is None:
            raise ValueError("Descriptor array is none")

        raw_matches = self.matcher.knnMatch(descriptors_a, descriptors_b, k=2)

        good_matches = []

        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue

            best, second_best= match_pair

            if best.distance < self.ratio_threshold * second_best.distance:
                good_matches.append(best)

        return good_matches

    
    def extract_point_pairs(self, keypoints_a: list, keypoints_b: list, good_matches: list) -> tuple:

        pts_a = np.float32(
            [keypoints_a[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        pts_b = np.float32(
            [keypoints_b[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        return pts_a, pts_b

    def visualize_matches(image_a: np.ndarray, keypoints_a: list, image_b: np.ndarray, keypoints_b: list, good_matches: list, detector_name: str ='SIFT', ratio_threshold: float = 0.7, max_display:int =100, output_path: str =None) -> None:

        display_matches = good_matches[:max_display]

        vis_img = cv2.drawMatches(image_a, keypoints_a, image_b, keypoints_b, display_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature Matches ({detector_name}, τ={ratio_threshold})")
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {output_path}")
        
        plt.show()

        return vis_img

if __name__ == "__main__":
    import sys

    DETECTOR = "SIFT"
    RATIO = 0.7

    path_a = "/Users/Antino/Desktop/Image Stitcher/images/left_room.jpg"
    path_b = "/Users/Antino/Desktop/Image Stitcher/images/right_room.jpg" 
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    if img_a is None:
        print(f"[ERROR] Could not load Image A from: {path_a}")
        sys.exit(1)
    if img_b is None:
        print(f"[ERROR] Could not load Image B from: {path_b}")
        sys.exit(1)

    matcher = FeatureMatcher(detector_type=DETECTOR, ratio_threshold=RATIO)

     print(f"[Step 1a] Detecting {DETECTOR} keypoints and computing descriptors...")
    kps_a, desc_a = matcher.detect_and_describe(img_a)
    kps_b, desc_b = matcher.detect_and_describe(img_b)
    print(f"          Image A → {len(kps_a)} keypoints")
    print(f"          Image B → {len(kps_b)} keypoints")

    print(f"[Step 1b] Matching descriptors with Lowe's Ratio Test (τ={RATIO})...")
    good = matcher.match(desc_a, desc_b)
    print(f"          Good matches retained: {len(good)}")

    print("[Step 1c] Extracting point pairs for Homography (Step 2 input)...")
    pts_a, pts_b = matcher.extract_point_pairs(kps_a, kps_b, good)
    print(f"          pts_a: {pts_a.shape}  |  pts_b: {pts_b.shape}")

    print("[Step 1d] Visualizing matches...")
    visualize_matches(img_a, kps_a, img_b, kps_b, good,
                      detector_name=DETECTOR, ratio_threshold=RATIO,
                      output_path="match_result.png")

    print(f"\n[✓] Step 1 Complete — pts_a, pts_b ready for Step 2 (Homography & RANSAC).")
