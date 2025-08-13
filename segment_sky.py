#!/usr/bin/env python3
"""
Sky segmentation script for processing aligned image series.

This script finds subdirectories with aligned images and applies sky segmentation
to each image, saving the results as PNG files.
"""

import argparse
from pathlib import Path
import OpenEXR
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2 as cv
import copy
from abc import ABC, abstractmethod


# Abstract Base Class for Sky Segmentation Strategies
class SkySegmentationStrategy(ABC):
    """Abstract base class for sky segmentation strategies."""

    @abstractmethod
    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        """
        Segment sky from the input image.

        Args:
            image_array (np.ndarray): Input image as numpy array (RGB format)

        Returns:
            np.ndarray: Sky mask as RGB image
        """
        pass


# Factory Singleton for Sky Segmentation Strategies
class SkySegmentationFactory:
    _instance = None
    _creators = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SkySegmentationFactory, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_creator(cls, key, creator):
        cls._creators[key] = creator

    @classmethod
    def get_segmenter(cls, strategy_type):
        creator = cls._creators.get(strategy_type)
        if not creator:
            available_strategies = ", ".join(cls._creators.keys())
            raise ValueError(
                f"Unknown segmentation strategy: {strategy_type}. Available: {available_strategies}"
            )
        return creator()

    @classmethod
    def get_available_strategies(cls):
        return list(cls._creators.keys())


# Strategy Implementations
class DummySegmentation(SkySegmentationStrategy):
    """Placeholder segmentation that returns the original image."""

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        """Return the input image unchanged."""
        return image_array


SkySegmentationFactory.register_creator("dummy", DummySegmentation)


class SkySegONNX(SkySegmentationStrategy):
    """Sky segmentation using the skyseg ONNX model."""

    def __init__(self, model_path="skyseg.onnx", input_size=(320, 320)):
        """
        Initialize SkySegONNX strategy.

        Args:
            model_path (str): Path to the skyseg ONNX model
            input_size (tuple): Input size for the model (width, height)
        """
        import onnxruntime

        self.model_path = model_path
        self.input_size = input_size
        try:
            self.onnx_session = onnxruntime.InferenceSession(self.model_path)
        except Exception as e:
            print(
                f"Model file {self.model_path} not found. Downloading from HuggingFace..."
            )
            import requests

            url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
            response = requests.get(url)
            with open(self.model_path, "wb") as f:
                f.write(response.content)
            self.onnx_session = onnxruntime.InferenceSession(self.model_path)

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        """
        Segment sky using skyseg ONNX model.

        Reference:
        https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/tree/main
        """
        try:
            # Convert RGB to BGR for OpenCV processing
            image_bgr = cv.cvtColor(image_array, cv.COLOR_RGB2BGR)

            # Downsample image if it's too large
            temp_image = copy.deepcopy(image_bgr)
            while temp_image.shape[0] >= 640 and temp_image.shape[1] >= 640:
                temp_image = cv.pyrDown(temp_image)

            # Run inference
            result_map = self._run_inference(temp_image)

            # Resize result back to original image size
            original_height, original_width = image_array.shape[:2]
            result_resized = cv.resize(result_map, (original_width, original_height))

            # Convert to 3-channel RGB for consistency
            result_rgb = cv.cvtColor(result_resized, cv.COLOR_GRAY2RGB)

            return result_rgb

        except Exception as e:
            print(f"Error in skyseg segmentation: {e}")
            return image_array

    def _run_inference(self, image):
        """Run inference with the skyseg ONNX model."""
        # Pre process: Resize, BGR->RGB, Transpose, PyTorch standardization
        temp_image = copy.deepcopy(image)
        resize_image = cv.resize(temp_image, dsize=self.input_size)
        x = cv.cvtColor(resize_image, cv.COLOR_BGR2RGB)
        x = np.array(x, dtype=np.float32)

        # PyTorch ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = (x / 255 - mean) / std
        x = x.transpose(2, 0, 1)
        x = x.reshape(-1, 3, self.input_size[0], self.input_size[1]).astype("float32")

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        onnx_result = self.onnx_session.run([output_name], {input_name: x})

        # Post process
        onnx_result = np.array(onnx_result).squeeze()
        min_value = np.min(onnx_result)
        max_value = np.max(onnx_result)
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
        onnx_result *= 255
        onnx_result = onnx_result.astype("uint8")

        return onnx_result


SkySegmentationFactory.register_creator("onnx", SkySegONNX)


class GroundedSAM2Segmentation(SkySegmentationStrategy):
    """Sky segmentation using GroundedSAM2 with sky prompt."""

    def __init__(self):
        """Initialize GroundedSAM2 strategy."""
        from autodistill_grounded_sam_2 import GroundedSAM2
        from autodistill.detection import CaptionOntology

        self.base_model = GroundedSAM2(
            ontology=CaptionOntology({"sky": "sky"}), model="Grounding DINO"
        )

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        """Segment sky using GroundedSAM2."""
        try:
            results = self.base_model.predict(image_array)
            mask = results.mask[0].astype(np.uint8) * 255
            mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

            return mask_rgb

        except Exception as e:
            print(f"Error in GroundedSAM2 segmentation: {e}")
            return image_array


SkySegmentationFactory.register_creator("groundedsam2", GroundedSAM2Segmentation)


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
            height, width = RGB.shape[0:2]

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
        method_name (str): Name of segmentation method to use
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    # Get segmentation function
    segmentation_func = SkySegmentationFactory.get_segmenter(method_name)

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
        output_dir_name = f"{subdir_name}_sky_segmented_{method_name}"
        output_dir = parent_dir / output_dir_name
        output_dir.mkdir(exist_ok=True)

        # Process each EXR file in the source directory
        exr_files = list(source_dir.glob("*.exr"))

        if not exr_files:
            continue

        # Use tqdm for progress bar
        for exr_file in tqdm(
            sorted(exr_files), desc=f"Processing {parent_dir.name}", unit="image"
        ):
            # Load EXR image
            image_array = load_exr_image(exr_file)
            if image_array is None:
                continue

            # Apply segmentation
            segmented_image = segmentation_func(image_array)

            # Create output filename (change extension to .png)
            output_filename = exr_file.stem + ".png"
            output_path = output_dir / output_filename

            # Save as PNG
            save_png_image(segmented_image, output_path)


def main():
    """Main function to parse arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Sky segmentation for aligned image series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-path",
        default="/work3/monka/lez_paper/example_data",
        help="Root path to search for image directories",
    )

    parser.add_argument(
        "--subdir-name",
        default="series_aligned_final",
        help="Name of subdirectory containing aligned images",
    )

    parser.add_argument("--method", default="dummy", help="Segmentation method to use")

    args = parser.parse_args()

    print("Sky Segmentation Script")
    print(f"Input path: {args.input_path}")
    print(f"Subdirectory name: {args.subdir_name}")
    print(f"Method: {args.method}")
    print("-" * 50)

    process_directory(args.input_path, args.subdir_name, args.method)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
