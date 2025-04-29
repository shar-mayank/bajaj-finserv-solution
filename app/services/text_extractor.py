"""
Text extraction service for lab report processing.

This module extracts and parses lab test data from OCR text.
"""

import re
import json
import pytesseract
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Pattern, Callable, Match
import cv2

from app.models.lab_test import LabTest

# Configure logging
logger = logging.getLogger(__name__)

# Common lab test patterns
LAB_TEST_PATTERNS = {
    # CBC (Complete Blood Count) patterns
    "cbc": [
        r"(?P<test_name>Haemoglobin|Hemoglobin|RBC|WBC|Platelet\s*Count|HCT|PCV|MCV|MCH|MCHC|RDW|TLC)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>[A-Za-z%/]+)?\s*(?P<bio_reference_range>[\d\.\-]+)",
        r"(?P<test_name>Neutrophils|Lymphocytes|Monocytes|Eosinophils|Basophils)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>[A-Za-z%/]+)?\s*(?P<bio_reference_range>[\d\.\-]+)",
    ],

    # Lipid Profile patterns
    "lipid": [
        r"(?P<test_name>Total\s*Cholesterol|HDL\s*Cholesterol|LDL\s*Cholesterol|Triglycerides|VLDL)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>mg/dL)?\s*(?P<bio_reference_range>[\d\.\-]+)",
    ],

    # Liver Function Test patterns
    "liver": [
        r"(?P<test_name>Total\s*Bilirubin|Direct\s*Bilirubin|SGOT|SGPT|Alkaline\s*Phosphatase|Total\s*Protein|Albumin|Globulin)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>[A-Za-z/]+)?\s*(?P<bio_reference_range>[\d\.\-]+)",
    ],

    # Kidney Function Test patterns
    "kidney": [
        r"(?P<test_name>Urea|Creatinine|Uric\s*Acid|BUN)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>[A-Za-z/]+)?\s*(?P<bio_reference_range>[\d\.\-]+)",
    ],

    # Glucose patterns
    "glucose": [
        r"(?P<test_name>Fasting\s*Glucose|Random\s*Glucose|HbA1c|Glucose|FBS|RBS)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>[A-Za-z%/]+)?\s*(?P<bio_reference_range>[\d\.\-]+)",
    ],

    # Thyroid patterns
    "thyroid": [
        r"(?P<test_name>T3|T4|TSH|Free\s*T3|Free\s*T4)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>[A-Za-z/]+)?\s*(?P<bio_reference_range>[\d\.\-]+)",
    ],

    # CRP pattern
    "crp": [
        r"(?P<test_name>C[\-\s]Reactive\s*Protein|CRP)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?:[\[\(](?P<flag>[HL])[\]\)])?\s*(?P<test_unit>mg[/]?[Ll]|mg/dL)?\s*(?:Biological\s*Ref\.\s*Range|Reference\s*Range)?\s*(?P<bio_reference_range>[\d\.\-<>]+)",
    ]
}

# General test pattern - more flexible but less specific
GENERAL_TEST_PATTERN = r"(?P<test_name>[A-Za-z\s\(\)]+)\s*:?\s*(?P<test_value>[\d\.]+)\s*(?:[\[\(](?P<flag>[HL])[\]\)])?\s*(?P<test_unit>[A-Za-z%/]+)?\s*(?P<bio_reference_range>[\d\.\-<>]+)"

def extract_lab_tests(processed_image: np.ndarray) -> List[LabTest]:
    """
    Extract lab tests from a processed image.

    Args:
        processed_image: The preprocessed image as a numpy array

    Returns:
        List of extracted LabTest objects
    """
    try:
        # Run OCR on the processed image
        ocr_text = pytesseract.image_to_string(processed_image)
        logger.info(f"OCR text length: {len(ocr_text)}")
        logger.info(f"OCR Text sample: {ocr_text[:200]}...")

        # Try different extraction strategies
        lab_tests = []

        # 1. Try to extract from JSON (if the report is in JSON format)
        json_tests = extract_json_data(ocr_text)
        if json_tests:
            logger.info(f"Extracted {len(json_tests)} tests from JSON format")
            lab_tests.extend(json_tests)
            return lab_tests

        # 2. Determine the report type for specialized extraction
        report_type = identify_report_type(ocr_text)
        logger.info(f"Identified report type: {report_type}")

        # 3. Extract based on report type
        if report_type:
            specialized_tests = extract_by_report_type(ocr_text, report_type)
            if specialized_tests:
                logger.info(f"Extracted {len(specialized_tests)} tests using specialized extraction")
                lab_tests.extend(specialized_tests)

        # 4. Apply general extraction as a fallback
        if not lab_tests:
            general_tests = extract_with_general_pattern(ocr_text)
            if general_tests:
                logger.info(f"Extracted {len(general_tests)} tests using general pattern")
                lab_tests.extend(general_tests)

        # 5. Try table-based extraction if other methods yielded few results
        if len(lab_tests) < 3:
            try:
                # Extract from tables (if present)
                table_tests = extract_from_tables(processed_image)
                if table_tests:
                    logger.info(f"Extracted {len(table_tests)} tests from tables")
                    lab_tests.extend(table_tests)
            except Exception as e:
                logger.warning(f"Error in table extraction: {str(e)}")

        # Remove duplicates by test name
        lab_tests = remove_duplicate_tests(lab_tests)

        # Log extraction results
        logger.info(f"Total tests extracted: {len(lab_tests)}")
        return lab_tests

    except Exception as e:
        logger.error(f"Error extracting lab tests: {str(e)}")
        return []

def identify_report_type(text: str) -> str:
    """
    Identify the type of lab report based on OCR text.

    Args:
        text: OCR text from the lab report

    Returns:
        Report type identifier string
    """
    # Keywords to identify report types
    report_types = {
        "cbc": ["COMPLETE BLOOD COUNT", "CBC", "HEMOGRAM", "BLOOD COUNT", "HAEMOGRAM"],
        "lipid": ["LIPID PROFILE", "LIPIDS", "CHOLESTEROL", "TRIGLYCERIDES"],
        "liver": ["LIVER FUNCTION", "LFT", "HEPATIC", "LIVER PANEL"],
        "kidney": ["KIDNEY FUNCTION", "RENAL FUNCTION", "KFT", "RENAL PANEL"],
        "glucose": ["GLUCOSE", "SUGAR", "HBA1C", "GLYCOSYLATED", "DIABETES"],
        "thyroid": ["THYROID", "TSH", "T3", "T4", "THYROXINE"],
        "crp": ["CRP", "C-REACTIVE PROTEIN", "REACTIVE PROTEIN"]
    }

    # Check for each report type
    for report_type, keywords in report_types.items():
        for keyword in keywords:
            if re.search(keyword, text, re.IGNORECASE):
                return report_type

    return "unknown"

def extract_json_data(text: str) -> List[LabTest]:
    """
    Extract lab tests from JSON format in the OCR text.

    Args:
        text: OCR text from the lab report

    Returns:
        List of extracted LabTest objects
    """
    try:
        # Try to extract JSON-like structure
        json_pattern = r'\{\s*"is_success":\s*true,\s*"data":\s*\[.*?\]\s*\}'
        json_match = re.search(json_pattern, text, re.DOTALL)

        if json_match:
            # Extract the JSON text and parse it
            json_text = json_match.group(0)
            # Replace all linebreaks and extra spaces
            json_text = re.sub(r'\s+', ' ', json_text)
            # Make sure all property names are in quotes
            json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)
            # Replace single quotes with double quotes
            json_text = json_text.replace("'", '"')

            data = json.loads(json_text)
            lab_tests = []

            for test_data in data.get("data", []):
                lab_test = LabTest.create(
                    test_name=test_data.get("test_name", ""),
                    test_value=test_data.get("test_value", ""),
                    bio_reference_range=test_data.get("bio_reference_range", ""),
                    test_unit=test_data.get("test_unit", ""),
                    lab_test_out_of_range=test_data.get("lab_test_out_of_range", None)
                )
                lab_tests.append(lab_test)

            return lab_tests
    except Exception as e:
        logger.error(f"Error extracting JSON data: {e}")

    return []

def extract_by_report_type(text: str, report_type: str) -> List[LabTest]:
    """
    Extract lab tests based on the identified report type.

    Args:
        text: OCR text from the lab report
        report_type: The identified report type

    Returns:
        List of extracted LabTest objects
    """
    lab_tests = []

    # Get patterns for the report type
    patterns = LAB_TEST_PATTERNS.get(report_type, [])

    # Apply each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                test_name = match.group("test_name").strip()
                test_value = match.group("test_value").strip()

                # Get optional groups with defaults
                bio_reference_range = match.group("bio_reference_range").strip() if "bio_reference_range" in match.groupdict() else ""
                test_unit = match.group("test_unit").strip() if "test_unit" in match.groupdict() and match.group("test_unit") else ""

                # Check for flag (H/L) indicating out of range
                flag = match.group("flag") if "flag" in match.groupdict() and match.group("flag") else None
                lab_test_out_of_range = True if flag else None  # None means calculate automatically

                lab_test = LabTest.create(
                    test_name=test_name,
                    test_value=test_value,
                    bio_reference_range=bio_reference_range,
                    test_unit=test_unit,
                    lab_test_out_of_range=lab_test_out_of_range
                )
                lab_tests.append(lab_test)
            except (IndexError, AttributeError) as e:
                logger.warning(f"Error processing match: {e}")

    return lab_tests

def extract_with_general_pattern(text: str) -> List[LabTest]:
    """
    Extract lab tests using a general pattern approach.

    Args:
        text: OCR text from the lab report

    Returns:
        List of extracted LabTest objects
    """
    lab_tests = []

    # Apply general pattern
    matches = re.finditer(GENERAL_TEST_PATTERN, text)
    for match in matches:
        try:
            test_name = match.group("test_name").strip()
            test_value = match.group("test_value").strip()

            # Get optional groups with defaults
            bio_reference_range = match.group("bio_reference_range").strip() if "bio_reference_range" in match.groupdict() else ""
            test_unit = match.group("test_unit").strip() if "test_unit" in match.groupdict() and match.group("test_unit") else ""

            # Check for flag (H/L) indicating out of range
            flag = match.group("flag") if "flag" in match.groupdict() and match.group("flag") else None
            lab_test_out_of_range = True if flag else None  # None means calculate automatically

            # Skip if test name contains prohibited words (likely false positives)
            prohibited_words = ["date", "time", "patient", "doctor", "test", "report", "page", "sample"]
            if any(word in test_name.lower() for word in prohibited_words):
                continue

            lab_test = LabTest.create(
                test_name=test_name,
                test_value=test_value,
                bio_reference_range=bio_reference_range,
                test_unit=test_unit,
                lab_test_out_of_range=lab_test_out_of_range
            )
            lab_tests.append(lab_test)
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error processing general match: {e}")

    return lab_tests

def extract_from_tables(image: np.ndarray) -> List[LabTest]:
    """
    Extract lab tests from tables in the image.

    Args:
        image: The preprocessed image

    Returns:
        List of extracted LabTest objects
    """
    from app.services.image_processor import enhance_image_for_table_detection

    # Enhanced image for table detection
    table_mask = enhance_image_for_table_detection(image)

    # Find contours in the table mask
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract text from each cell
    lab_tests = []

    # Process if contours are found (table is present)
    if contours:
        # Sort contours by y-coordinate (to get rows in order)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_boxes = sorted(bounding_boxes, key=lambda b: b[1])

        # Group boxes by rows (similar y-coordinate)
        rows = []
        current_row = [sorted_boxes[0]]
        row_height_threshold = image.shape[0] * 0.03  # 3% of image height

        for box in sorted_boxes[1:]:
            if abs(box[1] - current_row[0][1]) < row_height_threshold:
                current_row.append(box)
            else:
                rows.append(sorted(current_row, key=lambda b: b[0]))  # Sort by x-coordinate
                current_row = [box]

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        # Process rows to extract data
        if len(rows) >= 2:  # At least header and one data row
            # Extract headers and data
            header_row = rows[0]
            data_rows = rows[1:]

            # Extract header text
            headers = []
            for box in header_row:
                x, y, w, h = box
                cell_img = image[y:y+h, x:x+w]
                cell_text = pytesseract.image_to_string(cell_img, config='--psm 6').strip()
                headers.append(cell_text)

            # Map column indices to expected content
            name_idx = value_idx = range_idx = unit_idx = -1
            for i, header in enumerate(headers):
                header_lower = header.lower()
                if any(word in header_lower for word in ["test", "parameter", "investigation"]):
                    name_idx = i
                elif any(word in header_lower for word in ["result", "value"]):
                    value_idx = i
                elif any(word in header_lower for word in ["range", "reference", "normal"]):
                    range_idx = i
                elif any(word in header_lower for word in ["unit"]):
                    unit_idx = i

            # Process data rows
            for row in data_rows:
                try:
                    if len(row) < 2:  # Need at least test name and value
                        continue

                    # Extract cell text
                    cells = []
                    for box in row:
                        x, y, w, h = box
                        cell_img = image[y:y+h, x:x+w]
                        cell_text = pytesseract.image_to_string(cell_img, config='--psm 6').strip()
                        cells.append(cell_text)

                    # Extract test data based on identified columns
                    test_name = cells[name_idx] if 0 <= name_idx < len(cells) else ""
                    test_value = cells[value_idx] if 0 <= value_idx < len(cells) else ""
                    bio_range = cells[range_idx] if 0 <= range_idx < len(cells) else ""
                    test_unit = cells[unit_idx] if 0 <= unit_idx < len(cells) else ""

                    # Clean values
                    test_value_clean = re.search(r'[\d\.]+', test_value).group(0) if re.search(r'[\d\.]+', test_value) else ""

                    # Check for flags
                    out_of_range = None
                    if re.search(r'\[\s*[HL]\s*\]', test_value):
                        out_of_range = True

                    if test_name and test_value_clean:
                        lab_test = LabTest.create(
                            test_name=test_name,
                            test_value=test_value_clean,
                            bio_reference_range=bio_range,
                            test_unit=test_unit,
                            lab_test_out_of_range=out_of_range
                        )
                        lab_tests.append(lab_test)
                except Exception as e:
                    logger.warning(f"Error processing table row: {e}")

    return lab_tests

def remove_duplicate_tests(tests: List[LabTest]) -> List[LabTest]:
    """
    Remove duplicate tests based on test name.

    Args:
        tests: List of LabTest objects

    Returns:
        List of LabTest objects with duplicates removed
    """
    unique_tests = {}

    for test in tests:
        # Normalize test name for comparison
        normalized_name = test.test_name.lower().strip()

        # Keep only first occurrence or the one with more complete data
        if normalized_name not in unique_tests or has_more_data(test, unique_tests[normalized_name]):
            unique_tests[normalized_name] = test

    return list(unique_tests.values())

def has_more_data(test1: LabTest, test2: LabTest) -> bool:
    """
    Check if test1 has more complete data than test2.

    Args:
        test1: First LabTest object
        test2: Second LabTest object

    Returns:
        True if test1 has more complete data, False otherwise
    """
    # Count non-empty fields
    count1 = sum(1 for field in [test1.bio_reference_range, test1.test_unit] if field)
    count2 = sum(1 for field in [test2.bio_reference_range, test2.test_unit] if field)

    return count1 > count2