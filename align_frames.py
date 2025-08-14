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
import cv2 as cv
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
                    # Invert the homography to transform current image to reference coordinates
                    # H represents reference -> current, but we need current -> reference
                    H_inv = np.linalg.inv(H)

                    # Scale homography to original image size if needed
                    original_height, original_width = current_image.shape[:2]
                    if (
                        self.img_size != original_width
                        or self.img_size != original_height
                    ):
                        scale_x = original_width / self.img_size
                        scale_y = original_height / self.img_size

                        # Scale the inverted homography matrix
                        S = np.array(
                            [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]],
                            dtype=np.float32,
                        )
                        S_inv = np.array(
                            [[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1]],
                            dtype=np.float32,
                        )
                        H_scaled = S @ H_inv @ S_inv
                    else:
                        H_scaled = H_inv

                    # Apply homography transformation
                    aligned_bgr = self.cv.warpPerspective(
                        img_bgr,
                        H_scaled,
                        (original_width, original_height),
                        flags=self.cv.INTER_LINEAR,
                        # borderMode=self.cv.BORDER_CONSTANT,
                        # borderValue=0,
                        borderMode=self.cv.BORDER_REFLECT_101,
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


class SuperPointLG_MeshWarpAlignment(FrameAlignmentStrategy):
    """
    Frame alignment using SuperPoint-LG matcher with mesh-based warping.
    Uses feature correspondences to create a mesh deformation instead of global homography.
    """

    def __init__(self, device="cuda", img_size=512, mesh_density=20):
        """
        Initialize SuperPointLG Mesh Warp alignment strategy.

        Args:
            device (str): Device to use ('cuda' or 'cpu')
            img_size (int): Image size for processing
            mesh_density (int): Number of mesh grid points per dimension
        """
        from matching import get_matcher
        import cv2 as cv
        from scipy.interpolate import griddata

        self.device = device
        self.img_size = img_size
        self.mesh_density = mesh_density
        self.matcher = get_matcher("superpoint-lg", device=device)
        self.cv = cv
        self.griddata = griddata

    def _create_mesh_grid(self, height, width):
        """Create a regular mesh grid for warping."""
        y_coords = np.linspace(0, height - 1, self.mesh_density)
        x_coords = np.linspace(0, width - 1, self.mesh_density)
        mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
        return mesh_x.flatten(), mesh_y.flatten()

    def _interpolate_displacement(self, src_points, dst_points, mesh_x, mesh_y):
        """Interpolate displacement field from correspondences to mesh points."""
        if len(src_points) < 3:
            # Not enough points for interpolation, return identity mapping
            return mesh_x, mesh_y

        # Calculate displacement vectors
        displacement_x = dst_points[:, 0] - src_points[:, 0]
        displacement_y = dst_points[:, 1] - src_points[:, 1]

        # Interpolate displacements to mesh grid
        try:
            mesh_points = np.column_stack([mesh_x, mesh_y])

            # Use griddata for interpolation with fallback to nearest neighbor
            disp_x_interp = self.griddata(
                src_points, displacement_x, mesh_points, method="linear", fill_value=0.0
            )
            disp_y_interp = self.griddata(
                src_points, displacement_y, mesh_points, method="linear", fill_value=0.0
            )

            # Handle NaN values by filling with nearest neighbor
            nan_mask = np.isnan(disp_x_interp) | np.isnan(disp_y_interp)
            if np.any(nan_mask):
                disp_x_nn = self.griddata(
                    src_points, displacement_x, mesh_points, method="nearest"
                )
                disp_y_nn = self.griddata(
                    src_points, displacement_y, mesh_points, method="nearest"
                )
                disp_x_interp[nan_mask] = disp_x_nn[nan_mask]
                disp_y_interp[nan_mask] = disp_y_nn[nan_mask]

            # Apply displacements to get new mesh positions
            new_mesh_x = mesh_x + disp_x_interp
            new_mesh_y = mesh_y + disp_y_interp

            return new_mesh_x, new_mesh_y

        except Exception as e:
            print(f"Interpolation failed: {e}, using identity mapping")
            return mesh_x, mesh_y

    def _apply_mesh_warp(self, image, src_mesh_x, src_mesh_y, dst_mesh_x, dst_mesh_y):
        """Apply mesh-based warping using OpenCV's remap function."""
        height, width = image.shape[:2]

        # Create coordinate arrays for the full image
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Reshape mesh coordinates
        src_points = np.column_stack([src_mesh_x.flatten(), src_mesh_y.flatten()])
        dst_points = np.column_stack([dst_mesh_x.flatten(), dst_mesh_y.flatten()])

        # Add corner points to ensure full coverage
        corners_src = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        corners_dst = corners_src.copy()  # Keep corners fixed

        src_points = np.vstack([src_points, corners_src])
        dst_points = np.vstack([dst_points, corners_dst])

        try:
            # Use scipy's griddata to interpolate the displacement field across the entire image
            # Calculate displacements
            disp_x = dst_points[:, 0] - src_points[:, 0]
            disp_y = dst_points[:, 1] - src_points[:, 1]

            # Create coordinate grids for interpolation
            grid_points = np.column_stack([x_coords.flatten(), y_coords.flatten()])

            # Interpolate displacements to full image grid
            interp_disp_x = self.griddata(
                src_points, disp_x, grid_points, method="linear", fill_value=0.0
            ).reshape(height, width)

            interp_disp_y = self.griddata(
                src_points, disp_y, grid_points, method="linear", fill_value=0.0
            ).reshape(height, width)

            # Handle NaN values
            nan_mask = np.isnan(interp_disp_x) | np.isnan(interp_disp_y)
            if np.any(nan_mask):
                # Fill NaN values with nearest neighbor interpolation
                disp_x_nn = self.griddata(
                    src_points, disp_x, grid_points, method="nearest"
                ).reshape(height, width)
                disp_y_nn = self.griddata(
                    src_points, disp_y, grid_points, method="nearest"
                ).reshape(height, width)

                interp_disp_x[nan_mask] = disp_x_nn[nan_mask]
                interp_disp_y[nan_mask] = disp_y_nn[nan_mask]

            # Create the remapping coordinates
            map_x = (x_coords + interp_disp_x).astype(np.float32)
            map_y = (y_coords + interp_disp_y).astype(np.float32)

            # Apply the warping using OpenCV's remap
            warped = self.cv.remap(
                image,
                map_x,
                map_y,
                interpolation=self.cv.INTER_LINEAR,
                borderMode=self.cv.BORDER_REFLECT_101,
            )

            return warped

        except Exception as e:
            print(f"Mesh warping failed: {e}, returning original image")
            return image

    def __call__(
        self, reference_image: np.ndarray, current_image: np.ndarray
    ) -> np.ndarray:
        """
        Align current frame to reference using SuperPoint-LG matcher with mesh warping.

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

            # Create temporary files for the matcher
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

                # Extract matched keypoints
                if "matched_kpts0" in result and "matched_kpts1" in result:
                    matched_kpts0 = result["matched_kpts0"]  # Reference image keypoints
                    matched_kpts1 = result["matched_kpts1"]  # Current image keypoints
                    num_matches = len(matched_kpts0)

                    # print(f"Found {num_matches} feature matches for mesh warping")

                    if num_matches < 3:
                        print(
                            "Not enough matches for mesh warping, returning original image"
                        )
                        return current_image

                    # Scale keypoints to original image size
                    original_height, original_width = current_image.shape[:2]
                    if (
                        self.img_size != original_width
                        or self.img_size != original_height
                    ):
                        scale_x = original_width / self.img_size
                        scale_y = original_height / self.img_size

                        matched_kpts0 = matched_kpts0 * [scale_x, scale_y]
                        matched_kpts1 = matched_kpts1 * [scale_x, scale_y]

                    # Create mesh grid
                    mesh_x, mesh_y = self._create_mesh_grid(
                        original_height, original_width
                    )

                    # Interpolate displacement field from correspondences
                    # We want to warp current image to reference, so:
                    # src_points = current image keypoints, dst_points = reference image keypoints
                    new_mesh_x, new_mesh_y = self._interpolate_displacement(
                        matched_kpts1, matched_kpts0, mesh_x, mesh_y
                    )

                    # Apply mesh warping
                    aligned_bgr = self._apply_mesh_warp(
                        img_bgr,
                        mesh_x.reshape(self.mesh_density, self.mesh_density),
                        mesh_y.reshape(self.mesh_density, self.mesh_density),
                        new_mesh_x.reshape(self.mesh_density, self.mesh_density),
                        new_mesh_y.reshape(self.mesh_density, self.mesh_density),
                    )

                    # Convert back to RGB
                    aligned_rgb = self.cv.cvtColor(aligned_bgr, self.cv.COLOR_BGR2RGB)
                    return aligned_rgb

                else:
                    print("No feature matches found, returning original image")
                    return current_image

        except Exception as e:
            print(f"Error in SuperPointLG Mesh Warp alignment: {e}")
            return current_image


FrameAlignmentFactory.register_creator(
    "superpointlg_meshwarp", SuperPointLG_MeshWarpAlignment
)


class SuperPointLG_TPSAlignment(FrameAlignmentStrategy):
    """
    Frame alignment using SuperPoint-LG matcher with Thin Plate Spline (TPS) warping.
    Uses feature correspondences to create smooth elastic deformations.
    """

    def __init__(self, device="cuda", img_size=512):
        """
        Initialize SuperPointLG TPS alignment strategy.

        Args:
            device (str): Device to use ('cuda' or 'cpu')
            img_size (int): Image size for processing
        """
        from matching import get_matcher
        import cv2 as cv
        from scipy.interpolate import Rbf
        from skimage.transform import warp

        self.device = device
        self.img_size = img_size
        self.matcher = get_matcher("superpoint-lg", device=device)
        self.cv = cv
        self.Rbf = Rbf
        self.warp = warp

    def _thin_plate_spline_warp(self, img, source_points, target_points):
        """
        Applies a Thin Plate Spline (TPS) warp to an image.

        Args:
            img (np.array): The image to be warped.
            source_points (np.array): The original coordinates of the control points.
            target_points (np.array): The new coordinates of the control points.

        Returns:
            np.array: The warped image.
        """
        h, w = img.shape[:2]

        # Create grid of all pixel coordinates
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.vstack([x_coords.ravel(), y_coords.ravel()])

        try:
            # Use Rbf (Radial Basis Function) to create the TPS warp
            # This interpolates the deformation from the control points to the entire image.
            rbf_x = self.Rbf(
                source_points[:, 0],
                source_points[:, 1],
                target_points[:, 0],
                function="thin_plate",
            )
            rbf_y = self.Rbf(
                source_points[:, 0],
                source_points[:, 1],
                target_points[:, 1],
                function="thin_plate",
            )

            # Apply the deformation to the grid
            warped_grid_x = rbf_x(grid[0], grid[1]).reshape(h, w)
            warped_grid_y = rbf_y(grid[0], grid[1]).reshape(h, w)

            # Stack the deformed grid coordinates to use with skimage.warp
            tform = np.stack([warped_grid_y, warped_grid_x], axis=-1)

            # Use scikit-image's warp function to apply the transformation
            warped_image = self.warp(
                img, tform, output_shape=(h, w), preserve_range=True
            )

            # Convert back to uint8 if needed
            if warped_image.dtype != np.uint8:
                warped_image = (warped_image * 255).astype(np.uint8)

            return warped_image

        except Exception as e:
            print(f"TPS warping failed: {e}, returning original image")
            return img

    def __call__(
        self, reference_image: np.ndarray, current_image: np.ndarray
    ) -> np.ndarray:
        """
        Align current frame to reference using SuperPoint-LG matcher with TPS warping.

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

            # Create temporary files for the matcher
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

                # Extract matched keypoints
                if "matched_kpts0" in result and "matched_kpts1" in result:
                    matched_kpts0 = result["matched_kpts0"]  # Reference image keypoints
                    matched_kpts1 = result["matched_kpts1"]  # Current image keypoints
                    num_matches = len(matched_kpts0)

                    print(f"Found {num_matches} feature matches for TPS warping")

                    if num_matches < 3:
                        print(
                            "Not enough matches for TPS warping, returning original image"
                        )
                        return current_image

                    # Scale keypoints to original image size
                    original_height, original_width = current_image.shape[:2]
                    if (
                        self.img_size != original_width
                        or self.img_size != original_height
                    ):
                        scale_x = original_width / self.img_size
                        scale_y = original_height / self.img_size

                        matched_kpts0 = matched_kpts0 * [scale_x, scale_y]
                        matched_kpts1 = matched_kpts1 * [scale_x, scale_y]

                    # Add corner points to ensure stable warping at image boundaries
                    corners_current = np.array(
                        [
                            [0, 0],
                            [original_width - 1, 0],
                            [original_width - 1, original_height - 1],
                            [0, original_height - 1],
                        ],
                        dtype=np.float32,
                    )
                    corners_reference = corners_current.copy()  # Keep corners fixed

                    # Combine feature matches with corner constraints
                    source_points = np.vstack([matched_kpts1, corners_current])
                    target_points = np.vstack([matched_kpts0, corners_reference])

                    # Convert current image to float for TPS processing
                    current_float = current_image.astype(np.float32) / 255.0

                    # Apply TPS warping
                    aligned_float = self._thin_plate_spline_warp(
                        current_float, source_points, target_points
                    )

                    # Convert back to uint8
                    if aligned_float.dtype != np.uint8:
                        aligned_image = (np.clip(aligned_float, 0, 1) * 255).astype(
                            np.uint8
                        )
                    else:
                        aligned_image = aligned_float

                    return aligned_image

                else:
                    print("No feature matches found, returning original image")
                    return current_image

        except Exception as e:
            print(f"Error in SuperPointLG TPS alignment: {e}")
            return current_image


FrameAlignmentFactory.register_creator("superpointlg_tps", SuperPointLG_TPSAlignment)


class SuperPointLG_GridWarpAlignment(FrameAlignmentStrategy):
    """
    Frame alignment using SuperPoint-LG matcher with grid-based warping.
    Based on "Content-Preserving Warps for 3D Video Stabilization" (SIGGRAPH 2009).
    Uses a regular grid deformation with similarity constraints.
    """

    def __init__(self, device="cuda", img_size=512, grid_w=8, grid_h=6):
        """
        Initialize SuperPointLG Grid Warp alignment strategy.

        Args:
            device (str): Device to use ('cuda' or 'cpu')
            img_size (int): Image size for processing
            grid_w (int): Number of grid cells horizontally
            grid_h (int): Number of grid cells vertically
        """
        from matching import get_matcher
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve

        self.device = device
        self.img_size = img_size
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.matcher = get_matcher("superpoint-lg", device=device)
        self.cv = cv
        self.csr_matrix = csr_matrix
        self.spsolve = spsolve

    def _create_grid(self, width, height):
        """Create regular grid locations."""
        block_w = width / self.grid_w
        block_h = height / self.grid_h

        grid_loc = []
        for y in range(self.grid_h + 1):
            for x in range(self.grid_w + 1):
                grid_x = block_w * x
                grid_y = block_h * y

                # Adjust boundary points to be inside image
                if x == self.grid_w:
                    grid_x -= 1.1
                if y == self.grid_h:
                    grid_y -= 1.1

                grid_loc.append([grid_x, grid_y])

        return np.array(grid_loc), block_w, block_h

    def _get_grid_indices_and_weights(self, pt, block_w, block_h):
        """Get grid indices and bilinear weights for a point."""
        x, y = pt[0], pt[1]

        # Find grid cell
        gx = int(np.floor(x / block_w))
        gy = int(np.floor(y / block_h))

        # Clamp to valid range
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))

        # Grid indices (4 corners of the cell)
        # Layout: 1--2
        #         |  |
        #         4--3
        ind = np.array(
            [
                gy * (self.grid_w + 1) + gx,  # top-left
                gy * (self.grid_w + 1) + gx + 1,  # top-right
                (gy + 1) * (self.grid_w + 1) + gx + 1,  # bottom-right
                (gy + 1) * (self.grid_w + 1) + gx,  # bottom-left
            ]
        )

        # Bilinear weights
        xl = gx * block_w
        xh = (gx + 1) * block_w
        yl = gy * block_h
        yh = (gy + 1) * block_h

        w = np.array(
            [
                (xh - x) * (yh - y),  # top-left
                (x - xl) * (yh - y),  # top-right
                (x - xl) * (y - yl),  # bottom-right
                (xh - x) * (y - yl),  # bottom-left
            ]
        )

        # Normalize weights
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum

        return ind, w

    def _solve_grid_warp(self, grid_loc, pts1, pts2, width, height, block_w, block_h):
        """
        Solve for optimal grid deformation using content-preserving warps.
        Based on SIGGRAPH 2009 "Content-Preserving Warps for 3D Video Stabilization".
        """
        try:
            num_grid_points = len(grid_loc)
            num_correspondences = len(pts1)

            if num_correspondences < 4:
                print("Not enough correspondences for grid warping")
                return grid_loc.copy()

            # Filter correspondences to only include those inside the image
            valid_mask = (
                (pts2[:, 0] >= 0)
                & (pts2[:, 0] < width)
                & (pts2[:, 1] >= 0)
                & (pts2[:, 1] < height)
            )
            pts1_valid = pts1[valid_mask]
            pts2_valid = pts2[valid_mask]

            if len(pts1_valid) < 4:
                print("Not enough valid correspondences inside image bounds")
                return grid_loc.copy()

            # Build sparse linear system Ax = b
            # Variables: [x0, y0, x1, y1, ...] for each grid point
            num_vars = num_grid_points * 2

            # Data constraints (correspondence constraints)
            data_triplets = []
            data_rhs = []
            constraint_idx = 0

            w_data = 1.0

            for pt1, pt2 in zip(pts1_valid, pts2_valid):
                # Get grid cell indices and bilinear weights
                ind, w = self._get_grid_indices_and_weights(pt2, block_w, block_h)

                # X constraint: sum(w_i * x_i) = target_x
                for j in range(4):
                    if ind[j] < num_grid_points:
                        data_triplets.append(
                            (constraint_idx, ind[j] * 2, w_data * w[j])
                        )
                data_rhs.append(w_data * pt1[0])
                constraint_idx += 1

                # Y constraint: sum(w_i * y_i) = target_y
                for j in range(4):
                    if ind[j] < num_grid_points:
                        data_triplets.append(
                            (constraint_idx, ind[j] * 2 + 1, w_data * w[j])
                        )
                data_rhs.append(w_data * pt1[1])
                constraint_idx += 1

            # Similarity constraints (shape preservation)
            similarity_triplets = []
            similarity_rhs = []

            w_similarity = 10.0  # Similarity weight

            def get_local_coord(p1, p2, p3):
                """Get local coordinates of p1 relative to edge p2->p3."""
                axis1 = p3 - p2
                axis2 = np.array([-axis1[1], axis1[0]])  # perpendicular
                v = p1 - p2

                axis1_norm2 = np.dot(axis1, axis1)
                axis2_norm2 = np.dot(axis2, axis2)

                if axis1_norm2 > 1e-8 and axis2_norm2 > 1e-8:
                    u = np.dot(v, axis1) / axis1_norm2
                    v_coord = np.dot(v, axis2) / axis2_norm2
                    return np.array([u, v_coord])
                else:
                    return np.array([0.0, 0.0])

            # Add similarity constraints for interior grid points
            for y in range(1, self.grid_h):
                for x in range(1, self.grid_w):
                    center_idx = y * (self.grid_w + 1) + x

                    # Four neighboring pairs (following the paper's approach)
                    neighbors = [
                        (
                            y * (self.grid_w + 1) + x - 1,
                            (y - 1) * (self.grid_w + 1) + x,
                        ),  # left, top
                        (
                            (y - 1) * (self.grid_w + 1) + x,
                            y * (self.grid_w + 1) + x + 1,
                        ),  # top, right
                        (
                            y * (self.grid_w + 1) + x + 1,
                            (y + 1) * (self.grid_w + 1) + x,
                        ),  # right, bottom
                        (
                            (y + 1) * (self.grid_w + 1) + x,
                            y * (self.grid_w + 1) + x - 1,
                        ),  # bottom, left
                    ]

                    for gid0, gid1 in neighbors:
                        if gid0 < num_grid_points and gid1 < num_grid_points:
                            # Compute reference local coordinates in original grid
                            ref_uv = get_local_coord(
                                grid_loc[center_idx], grid_loc[gid0], grid_loc[gid1]
                            )

                            # X coordinate constraint:
                            # x_center - x_gid0 - ref_uv[0] * (x_gid1 - x_gid0) - ref_uv[1] * (y_gid1 - y_gid0) = 0
                            similarity_triplets.append(
                                (constraint_idx, center_idx * 2, w_similarity)
                            )
                            similarity_triplets.append(
                                (
                                    constraint_idx,
                                    gid0 * 2,
                                    -w_similarity * (1 - ref_uv[0]),
                                )
                            )
                            similarity_triplets.append(
                                (constraint_idx, gid1 * 2, -w_similarity * ref_uv[0])
                            )
                            similarity_triplets.append(
                                (
                                    constraint_idx,
                                    gid0 * 2 + 1,
                                    -w_similarity * ref_uv[1],
                                )
                            )
                            similarity_triplets.append(
                                (constraint_idx, gid1 * 2 + 1, w_similarity * ref_uv[1])
                            )
                            similarity_rhs.append(0.0)
                            constraint_idx += 1

                            # Y coordinate constraint:
                            # y_center - y_gid0 - ref_uv[0] * (y_gid1 - y_gid0) + ref_uv[1] * (x_gid1 - x_gid0) = 0
                            similarity_triplets.append(
                                (constraint_idx, center_idx * 2 + 1, w_similarity)
                            )
                            similarity_triplets.append(
                                (
                                    constraint_idx,
                                    gid0 * 2 + 1,
                                    -w_similarity * (1 - ref_uv[0]),
                                )
                            )
                            similarity_triplets.append(
                                (
                                    constraint_idx,
                                    gid1 * 2 + 1,
                                    -w_similarity * ref_uv[0],
                                )
                            )
                            similarity_triplets.append(
                                (constraint_idx, gid0 * 2, w_similarity * ref_uv[1])
                            )
                            similarity_triplets.append(
                                (constraint_idx, gid1 * 2, -w_similarity * ref_uv[1])
                            )
                            similarity_rhs.append(0.0)
                            constraint_idx += 1

            # Combine all constraints
            all_triplets = data_triplets + similarity_triplets
            all_rhs = np.array(data_rhs + similarity_rhs)

            if len(all_triplets) == 0:
                print("No valid constraints generated")
                return grid_loc.copy()

            # Build sparse matrix
            rows, cols, values = zip(*all_triplets)
            A = self.csr_matrix((values, (rows, cols)), shape=(len(all_rhs), num_vars))

            # Solve the linear system using least squares
            try:
                # Use normal equations: A^T A x = A^T b
                AtA = A.T @ A
                Atb = A.T @ all_rhs
                solution = self.spsolve(AtA, Atb)

                # Extract grid coordinates
                new_grid = np.zeros_like(grid_loc)
                for i in range(num_grid_points):
                    new_grid[i, 0] = solution[i * 2]
                    new_grid[i, 1] = solution[i * 2 + 1]

                return new_grid

            except Exception as solve_error:
                print(f"Linear system solving failed: {solve_error}")
                # Fallback: use simple homography on grid points
                H, _ = self.cv.findHomography(
                    pts2_valid.astype(np.float32),
                    pts1_valid.astype(np.float32),
                    method=self.cv.RANSAC,
                )

                if H is not None:
                    # Apply homography to grid points
                    grid_homog = np.column_stack([grid_loc, np.ones(len(grid_loc))])
                    warped_grid_homog = (H @ grid_homog.T).T
                    new_grid = np.zeros_like(grid_loc)
                    new_grid[:, 0] = warped_grid_homog[:, 0] / warped_grid_homog[:, 2]
                    new_grid[:, 1] = warped_grid_homog[:, 1] / warped_grid_homog[:, 2]
                    return new_grid
                else:
                    return grid_loc.copy()

        except Exception as e:
            print(f"Grid warp solving failed: {e}")
            return grid_loc.copy()

    def _apply_grid_warp(self, image, original_grid, warped_grid, block_w, block_h):
        """
        Apply grid warping to image using bilinear interpolation.
        Following the SIGGRAPH paper's approach.
        """
        height, width = image.shape[:2]

        # Create maps for OpenCV remap
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        # For each pixel in the output image, find where it should sample from
        for y in range(height):
            for x in range(width):
                # Get grid indices and weights for current output pixel
                ind, w = self._get_grid_indices_and_weights(
                    np.array([x, y]), block_w, block_h
                )

                # Compute source position using bilinear interpolation on warped grid
                src_x = 0.0
                src_y = 0.0
                for i in range(4):
                    if ind[i] < len(warped_grid):
                        src_x += warped_grid[ind[i], 0] * w[i]
                        src_y += warped_grid[ind[i], 1] * w[i]

                # Store in remap arrays
                map_x[y, x] = src_x
                map_y[y, x] = src_y

        # Apply the warping using OpenCV's remap
        warped = self.cv.remap(
            image,
            map_x,
            map_y,
            interpolation=self.cv.INTER_LINEAR,
            borderMode=self.cv.BORDER_REFLECT_101,
        )

        return warped

    def __call__(
        self, reference_image: np.ndarray, current_image: np.ndarray
    ) -> np.ndarray:
        """
        Align current frame to reference using SuperPoint-LG matcher with grid warping.

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

            # Create temporary files for the matcher
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

                # Extract matched keypoints
                if "matched_kpts0" in result and "matched_kpts1" in result:
                    matched_kpts0 = result["matched_kpts0"]  # Reference image keypoints
                    matched_kpts1 = result["matched_kpts1"]  # Current image keypoints
                    num_matches = len(matched_kpts0)

                    print(f"Found {num_matches} feature matches for grid warping")

                    if num_matches < 4:
                        print(
                            "Not enough matches for grid warping, returning original image"
                        )
                        return current_image

                    # Scale keypoints to original image size
                    original_height, original_width = current_image.shape[:2]
                    if (
                        self.img_size != original_width
                        or self.img_size != original_height
                    ):
                        scale_x = original_width / self.img_size
                        scale_y = original_height / self.img_size

                        matched_kpts0 = matched_kpts0 * [scale_x, scale_y]
                        matched_kpts1 = matched_kpts1 * [scale_x, scale_y]

                    # Create regular grid
                    grid_loc, block_w, block_h = self._create_grid(
                        original_width, original_height
                    )

                    # Solve for optimal grid deformation
                    # pts1 = reference positions (where we want to map to)
                    # pts2 = current positions (where we map from)
                    warped_grid = self._solve_grid_warp(
                        grid_loc,
                        matched_kpts0,
                        matched_kpts1,
                        original_width,
                        original_height,
                        block_w,
                        block_h,
                    )

                    # Apply grid warping
                    aligned_image = self._apply_grid_warp(
                        current_image, grid_loc, warped_grid, block_w, block_h
                    )

                    return aligned_image

                else:
                    print("No feature matches found, returning original image")
                    return current_image

        except Exception as e:
            print(f"Error in SuperPointLG Grid Warp alignment: {e}")
            return current_image


FrameAlignmentFactory.register_creator(
    "superpointlg_gridwarp", SuperPointLG_GridWarpAlignment
)


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
                reference_image = (
                    image_array.copy()
                )  # Store first frame as reference for all subsequent frames
                print(f"First frame ({exr_file.name}) - copying as reference")
            else:
                # Apply alignment for subsequent frames using the first frame as reference
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
