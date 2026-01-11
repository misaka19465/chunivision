"""
Example demonstrating the TransformCalculator usage.

This example shows how to use the TransformCalculator to:
1. Calculate perspective transforms from calibration points
2. Apply transforms to new points
3. Validate transform quality
4. Use inverse transforms
"""

import numpy as np
from chunivision.calibration import TransformCalculator


def main():
    print("=" * 60)
    print("TransformCalculator Example")
    print("=" * 60)

    # Example 1: Basic perspective transform
    print("\n1. Basic Perspective Transform")
    print("-" * 60)

    # Define 4 corners of a quadrilateral in image coordinates
    image_points = np.array(
        [
            [100, 100],  # Top-left
            [500, 120],  # Top-right (slightly angled)
            [480, 380],  # Bottom-right
            [120, 360],  # Bottom-left
        ],
        dtype=np.float32,
    )

    # Corresponding world coordinates (e.g., calibration board)
    # 44cm x 9cm board
    world_points = np.array(
        [
            [0, 0],  # Top-left
            [44, 0],  # Top-right
            [44, 9],  # Bottom-right
            [0, 9],  # Bottom-left
        ],
        dtype=np.float32,
    )

    # Calculate the perspective transform
    transform = TransformCalculator.calculate_perspective_transform(
        image_points, world_points
    )

    print(f"Image points:\n{image_points}")
    print(f"\nWorld points:\n{world_points}")
    print(f"\nTransform matrix:\n{transform}")

    # Example 2: Apply transform to new points
    print("\n2. Apply Transform to New Points")
    print("-" * 60)

    # Test point in image coordinates (e.g., center of image)
    test_image_point = np.array([300, 240], dtype=np.float32)

    # Transform to world coordinates
    world_point = TransformCalculator.apply_transform(test_image_point, transform)

    print(f"Image point: {test_image_point}")
    print(f"World point: {world_point}")
    print(f"Position on board: {world_point[0]:.2f}cm, {world_point[1]:.2f}cm")

    # Example 3: Transform multiple points
    print("\n3. Transform Multiple Points")
    print("-" * 60)

    test_points = np.array(
        [[200, 200], [300, 200], [400, 200], [300, 300]], dtype=np.float32
    )

    world_points_batch = TransformCalculator.apply_transform(test_points, transform)

    for i, (img_pt, world_pt) in enumerate(zip(test_points, world_points_batch)):
        print(
            f"Point {i+1}: ({img_pt[0]:.0f}, {img_pt[1]:.0f}) -> "
            f"({world_pt[0]:.2f}cm, {world_pt[1]:.2f}cm)"
        )

    # Example 4: Inverse transform
    print("\n4. Inverse Transform (World -> Image)")
    print("-" * 60)

    # Calculate inverse transform
    inverse_transform = TransformCalculator.calculate_inverse_transform(transform)

    # Pick a world coordinate
    target_world = np.array([22, 4.5], dtype=np.float32)  # Center of board

    # Find corresponding image coordinate
    image_coord = TransformCalculator.apply_transform(target_world, inverse_transform)

    print(f"Target world position: {target_world} cm")
    print(f"Corresponding image position: {image_coord}")

    # Verify round-trip
    back_to_world = TransformCalculator.apply_transform(image_coord, transform)
    print(f"Round-trip back to world: {back_to_world}")
    print(f"Error: {np.linalg.norm(back_to_world - target_world):.6f} cm")

    # Example 5: Validate transform quality
    print("\n5. Validate Transform Quality")
    print("-" * 60)

    # Validate using original calibration points
    quality, mean_error, max_error = TransformCalculator.validate_transform_quality(
        transform, image_points, world_points, max_error_threshold=1.0
    )

    print(f"Quality score: {quality:.4f} (1.0 = perfect)")
    print(f"Mean error: {mean_error:.6f} cm")
    print(f"Max error: {max_error:.6f} cm")

    if quality > 0.99:
        print("✓ Transform is excellent!")
    elif quality > 0.9:
        print("✓ Transform is good")
    elif quality > 0.7:
        print("⚠ Transform is acceptable but could be better")
    else:
        print("✗ Transform quality is poor - recalibration recommended")

    # Example 6: Reprojection error
    print("\n6. Reprojection Error")
    print("-" * 60)

    reprojection_error = TransformCalculator.estimate_reprojection_error(
        transform, image_points, world_points
    )

    print(f"Reprojection error: {reprojection_error:.6f} cm")
    print(f"This is the average error when mapping calibration points")

    # Example 7: Decompose transform
    print("\n7. Decompose Transform")
    print("-" * 60)

    components = TransformCalculator.decompose_transform(transform)

    print(f"Translation: {components['translation']}")
    print(f"Scale: {components['scale']}")
    print(
        f"Rotation: {components['rotation']:.4f} radians "
        f"({np.degrees(components['rotation']):.2f} degrees)"
    )
    print(f"Shear: {components['shear']:.4f}")
    print(f"Perspective: {components['perspective']}")

    # Example 8: Realistic calibration workflow
    print("\n8. Realistic Calibration Workflow")
    print("-" * 60)

    # Simulate collecting calibration points from user clicks
    print("Simulating calibration point collection...")

    # User clicks 4 corners of calibration board in camera view
    user_clicks = np.array(
        [
            [150, 120],  # Top-left corner
            [490, 130],  # Top-right corner
            [475, 355],  # Bottom-right corner
            [165, 345],  # Bottom-left corner
        ],
        dtype=np.float32,
    )

    # Known physical dimensions of calibration board
    board_size = (44.0, 9.0)  # width, height in cm
    known_corners = np.array(
        [
            [0, 0],
            [board_size[0], 0],
            [board_size[0], board_size[1]],
            [0, board_size[1]],
        ],
        dtype=np.float32,
    )

    # Calculate calibration transform
    calib_transform = TransformCalculator.calculate_perspective_transform(
        user_clicks, known_corners
    )

    # Validate calibration
    quality, mean_err, max_err = TransformCalculator.validate_transform_quality(
        calib_transform, user_clicks, known_corners, max_error_threshold=0.5
    )

    print(f"Calibration quality: {quality:.4f}")
    print(f"Mean error: {mean_err:.4f} cm")
    print(f"Max error: {max_err:.4f} cm")

    if quality > 0.95:
        print("✓ Calibration successful! Ready to use.")

        # Now we can map any point in the camera view to real-world coordinates
        test_click = np.array([320, 240], dtype=np.float32)  # Image center
        real_position = TransformCalculator.apply_transform(test_click, calib_transform)
        print(
            f"\nExample: Click at {test_click} maps to "
            f"position ({real_position[0]:.2f}, {real_position[1]:.2f}) cm"
        )
    else:
        print("✗ Calibration quality is poor. Please recalibrate.")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
