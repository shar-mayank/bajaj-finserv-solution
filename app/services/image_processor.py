"""
Image processing service for lab report processing.

This module provides functions for preprocessing and enhancing lab report images
to improve OCR accuracy.
"""

import cv2
import numpy as np
import logging
from typing import Union, Tuple, Optional
import io
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.

    Args:
        image_data: Raw image data as bytes

    Returns:
        Processed image as a numpy array
    """
    try:
        # Convert the image to a format OpenCV can use
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Failed to decode image")
            raise ValueError("Invalid image data")

        # Apply preprocessing steps
        processed_img = apply_preprocessing_pipeline(img)

        return processed_img

    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise

def apply_preprocessing_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Apply a series of image preprocessing steps.

    Args:
        img: The input image as a numpy array

    Returns:
        Processed image as a numpy array
    """
    try:
        # Store original dimensions
        original_height, original_width = img.shape[:2]

        # Resize for processing if too large
        img = resize_image_if_needed(img)

        # Apply adaptive thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        # Invert binary image (black text on white background)
        binary = cv2.bitwise_not(binary)

        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Deskew
        deskewed = deskew_image(cleaned)

        return deskewed

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        # Return original grayscale if processing fails
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_image_if_needed(img: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    """
    Resize an image if it's too large while maintaining aspect ratio.

    Args:
        img: The input image
        max_dimension: Maximum dimension (width or height)

    Returns:
        Resized image if necessary, original image otherwise
    """
    height, width = img.shape[:2]

    # Check if resize is needed
    if max(height, width) <= max_dimension:
        return img

    # Calculate new dimensions
    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))

    # Resize
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    return resized

def deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Deskew (straighten) an image based on text orientation.

    Args:
        img: The input image

    Returns:
        Deskewed image
    """
    try:
        # Calculate skew angle
        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Skip if angle is negligible
        if abs(angle) < 0.1:
            return img

        # Rotate image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        logger.info(f"Deskewed image by {angle:.2f} degrees")
        return rotated

    except Exception as e:
        logger.warning(f"Failed to deskew image: {str(e)}")
        return img  # Return original if deskewing fails

def enhance_image_for_table_detection(img: np.ndarray) -> np.ndarray:
    """
    Enhance an image specifically for table detection.

    Args:
        img: The input image

    Returns:
        Enhanced image optimized for table detection
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Apply morphological operations to enhance horizontal and vertical lines
    kernel_h = np.ones((1, 50), np.uint8)
    kernel_v = np.ones((50, 1), np.uint8)

    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

    # Combine horizontal and vertical lines
    table_mask = cv2.bitwise_or(horizontal, vertical)

    return table_mask

def is_table_present(img: np.ndarray) -> bool:
    """
    Detect if a table is present in the image.

    Args:
        img: The input image

    Returns:
        True if a table is detected, False otherwise
    """
    # Enhanced image for table detection
    table_mask = enhance_image_for_table_detection(img)

    # Count non-zero pixels in the table mask
    line_pixels = cv2.countNonZero(table_mask)

    # Calculate the percentage of line pixels
    total_pixels = img.shape[0] * img.shape[1]
    line_percentage = (line_pixels / total_pixels) * 100

    # If more than 0.5% of pixels are part of lines, consider it a table
    return line_percentage > 0.5