#!/usr/bin/env python3
"""
Frame alignment script for processing image series.

This script finds subdirectories with images and applies frame alignment
to each image, saving the results as PNG files.
"""

import argparse
from pathlib import Path
import OpenEXR
import numpy as np
from PIL import Image
from tqdm import tqdm
from abc import ABC, abstractmethod


# Abstract Base Class for Frame Alignment Strategies
class FrameAlignmentStrategy(ABC):
    """Abstract base class for frame alignment strategies."""

    @abstractmethod
    def __call__(
        self, reference_image: np.ndarray, current_image: np.ndarray
    ) -> np.ndarray:
        """
        Align current frame to the reference image.

        Args:
            reference_image (np.ndarray): Reference image as numpy array (RGB format)
            current_image (np.ndarray): Current image to align as numpy array (RGB format)

        Returns:
            np.ndarray: Aligned image as RGB image
        """
        pass


# Factory Singleton for Frame Alignment Strategies
class FrameAlignmentFactory:
    _instance = None
    _creators = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FrameAlignmentFactory, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_creator(cls, key, creator):
        cls._creators[key] = creator

    @classmethod
    def get_aligner(cls, strategy_type):
        creator = cls._creators.get(strategy_type)
        if not creator:
            available_strategies = ", ".join(cls._creators.keys())
            raise ValueError(
                f"Unknown alignment strategy: {strategy_type}. Available: {available_strategies}"
            )
        return creator()

    @classmethod
    def get_available_strategies(cls):
        return list(cls._creators.keys())


# Strategy Implementations
class DummyAlignment(FrameAlignmentStrategy):
    """Placeholder alignment that returns the current image unchanged."""

    def __call__(
        self, reference_image: np.ndarray, current_image: np.ndarray
    ) -> np.ndarray:
        """Return the current image unchanged."""
        return current_image


FrameAlignmentFactory.register_creator("dummy", DummyAlignment)


class SuperPointLGAlignment(FrameAlignmentStrategy):
    """
    Frame alignment using SuperPoint-LG matcher with homography estimation.
    Nice benchmark: https://huggingface.co/spaces/Realcat/image-matching-webui
    """

    def __init__(self, device="cuda", img_size=512):
        """
        Initialize SuperPointLG alignment strategy.

        Args:
            device (str): Device to use ('cuda' or 'cpu')
            img_size (int): Image size for processing
        """
        from matching import get_matcher
        import cv2 as cv

        self.device = device
        self.img_size = img_size
        self.matcher = get_matcher("superpoint-lg", device=device)
        self.cv = cv

    def __call__(
        self, reference_image: np.ndarray, current_image: np.ndarray
    ) -> np.ndarray:
        """
        Align current frame to reference using SuperPoint-LG matcher.

        Args:
            reference_image (np.ndarray): Reference image as numpy array (RGB format)
            current_image (np.ndarray): Current image to align as numpy array (RGB format)

        Returns:
            np.ndarray: Aligned image as RGB image
        """
        try:
            # Convert RGB to BGR for OpenCV processing
            img_bgr = self.cv.cvtColor(current_image, self.cv.COLOR_RGB2BGR)
            ref_bgr = self.cv.cvtColor(reference_image, self.cv.COLOR_RGB2BGR)

            # Create temporary files for the matcher (if load_image_from_array doesn't exist)
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as temp_dir:
                ref_path = os.path.join(temp_dir, "ref.png")
                img_path = os.path.join(temp_dir, "img.png")

                # Save temporary images
                self.cv.imwrite(ref_path, ref_bgr)
                self.cv.imwrite(img_path, img_bgr)

                # Load images using the matcher
                img0 = self.matcher.load_image(ref_path, resize=self.img_size)
                img1 = self.matcher.load_image(img_path, resize=self.img_size)

                # Perform matching
                result = self.matcher(img0, img1)

                # Extract homography
                H = result.get("H")
                num_inliers = result.get("num_inliers", 0)

                # Alternative: Use PoseLib for more robust homography estimation
                if (
                    H is None
                    and "matched_kpts0" in result
                    and "matched_kpts1" in result
                ):
                    try:
                        import poselib

                        matched_kpts0 = result["matched_kpts0"]
                        matched_kpts1 = result["matched_kpts1"]

                        if (
                            len(matched_kpts0) >= 4
                        ):  # Need at least 4 points for homography
                            # Estimate homography using PoseLib
                            pose, info = poselib.estimate_homography(
                                matched_kpts0,
                                matched_kpts1,
                                {"max_reproj_error": 4.0, "max_iterations": 10000},
                            )

                            if pose is not None:
                                H = pose
                                num_inliers = info.get("num_inliers", 0)
                                print(
                                    f"PoseLib homography estimation: {num_inliers} inliers"
                                )

                    except ImportError:
                        print("PoseLib not available, using matcher result")
                    except Exception as e:
                        print(f"PoseLib estimation failed: {e}")

                # If we have a valid homography and enough inliers, apply it
                if H is not None and num_inliers > 10:
                    # Scale homography to original image size if needed
                    original_height, original_width = current_image.shape[:2]
                    if (
                        self.img_size != original_width
                        or self.img_size != original_height
                    ):
                        scale_x = original_width / self.img_size
                        scale_y = original_height / self.img_size

                        # Scale the homography matrix
                        S = np.array(
                            [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]],
                            dtype=np.float32,
                        )
                        S_inv = np.array(
                            [[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1]],
                            dtype=np.float32,
                        )
                        H_scaled = S @ H @ S_inv
                    else:
                        H_scaled = H

                    # Apply homography transformation
                    aligned_bgr = self.cv.warpPerspective(
                        img_bgr,
                        H_scaled,
                        (original_width, original_height),
                        flags=self.cv.INTER_LINEAR,
                        borderMode=self.cv.BORDER_CONSTANT,
                        borderValue=0,
                    )

                    # Convert back to RGB
                    aligned_rgb = self.cv.cvtColor(aligned_bgr, self.cv.COLOR_BGR2RGB)
                    return aligned_rgb

                else:
                    print(
                        f"Warning: Poor alignment quality (inliers: {num_inliers}), returning original image"
                    )
                    return current_image

        except Exception as e:
            print(f"Error in SuperPointLG alignment: {e}")
            return current_image


FrameAlignmentFactory.register_creator("superpointlg", SuperPointLGAlignment)


def load_exr_image(file_path, exposure=1.0, gamma=2.2):
    """
    Load an EXR image and convert to numpy array.

    Args:
        file_path (str): Path to the EXR file
        exposure (float): Exposure multiplier for tone mapping
        gamma (float): Gamma correction value

    Returns:
        np.ndarray: Image as numpy array
    """
    try:
        # Open the EXR file using the newer API
        with OpenEXR.File(str(file_path)) as infile:
            # Get RGB channels
            RGB = infile.channels()["RGB"].pixels

            # RGB is already a numpy array with shape (height, width, 3)
            linear_rgb = RGB.copy()

            # Apply exposure
            img = linear_rgb * exposure
            # Clamp to [0, 1]
            img = np.clip(img, 0, 1)
            # Gamma correction
            img = np.power(img, 1.0 / gamma)

            # Convert to 8-bit for PNG output
            rgb = (img * 255).astype(np.uint8)

            return rgb

    except Exception as e:
        print(f"Error loading EXR file {file_path}: {e}")
        return None


def save_png_image(image_array, output_path):
    """
    Save numpy array as PNG image.

    Args:
        image_array (np.ndarray): Image as numpy array
        output_path (str): Output file path
    """
    try:
        if image_array is not None:
            # Convert to PIL Image and save
            img = Image.fromarray(image_array)
            img.save(output_path)
            # print(f"Saved: {output_path}")
        else:
            print(f"Failed to save: {output_path} - image array is None")
    except Exception as e:
        print(f"Error saving PNG file {output_path}: {e}")


def process_directory(input_path, subdir_name, method_name):
    """
    Process all subdirectories containing the specified subdir_name.

    Args:
        input_path (str): Root path to search for subdirectories
        subdir_name (str): Name of subdirectory containing images
        method_name (str): Name of alignment method to use
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    # Get alignment function
    alignment_func = FrameAlignmentFactory.get_aligner(method_name)

    # Find all subdirectories containing the specified subdir_name
    target_dirs = []
    for item in input_path.iterdir():
        if item.is_dir():
            subdir_path = item / subdir_name
            if subdir_path.exists() and subdir_path.is_dir():
                target_dirs.append((item, subdir_path))

    if not target_dirs:
        # print(f"No directories found containing '{subdir_name}' in {input_path}")
        return

    # print(f"Found {len(target_dirs)} directories to process:")
    # for parent_dir, _ in target_dirs:
    #     print(f"  - {parent_dir.name}")

    # Process each directory
    for parent_dir, source_dir in target_dirs:
        # Create output directory at the same level
        output_dir_name = f"{subdir_name}_aligned_{method_name}"
        output_dir = parent_dir / output_dir_name
        output_dir.mkdir(exist_ok=True)

        # Process each EXR file in the source directory
        exr_files = list(source_dir.glob("*.exr"))

        if not exr_files:
            continue

        # For alignment strategies that need a reference image,
        # create a fresh instance for each directory
        if method_name == "superpointlg":
            alignment_func = FrameAlignmentFactory.get_aligner(method_name)

        # Use tqdm for progress bar
        reference_image = None
        for i, exr_file in enumerate(
            tqdm(sorted(exr_files), desc=f"Processing {parent_dir.name}", unit="image")
        ):
            # Load EXR image
            image_array = load_exr_image(exr_file)
            if image_array is None:
                continue

            # For the first frame, just copy it without alignment (it becomes the reference)
            if i == 0:
                aligned_image = image_array
                reference_image = image_array  # Store reference for subsequent frames
                print(f"First frame ({exr_file.name}) - copying as reference")
            else:
                # Apply alignment for subsequent frames
                aligned_image = alignment_func(reference_image, image_array)

            # Create output filename (change extension to .png)
            output_filename = exr_file.stem + ".png"
            output_path = output_dir / output_filename

            # Save as PNG
            save_png_image(aligned_image, output_path)


def main():
    """Main function to parse arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Frame alignment for image series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-path",
        default="/work3/monka/lez_paper/example_data",
        help="Root path to search for image directories",
    )

    parser.add_argument(
        "--subdir-name",
        default="series",
        help="Name of subdirectory containing images",
    )

    parser.add_argument("--method", default="dummy", help="Alignment method to use")

    args = parser.parse_args()

    print("Frame Alignment Script")
    print(f"Input path: {args.input_path}")
    print(f"Subdirectory name: {args.subdir_name}")
    print(f"Method: {args.method}")
    print("-" * 50)

    process_directory(args.input_path, args.subdir_name, args.method)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
