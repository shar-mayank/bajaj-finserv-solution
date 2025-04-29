"""
Helper functions for the lab report processing application.
"""
import re
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text from lab reports.

    Args:
        text: Raw text extracted from lab report.

    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove special characters but keep essential ones
    text = re.sub(r'[^\w\s.-:/]', '', text)

    return text

def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract numeric value from text string.

    Args:
        text: String that may contain a numeric value.

    Returns:
        Float value if found, None otherwise.
    """
    if not text:
        return None

    # Handle common variations of numeric formats
    # Remove any non-numeric characters except decimal points and minus signs
    numeric_pattern = r'[-+]?\d*\.?\d+'
    match = re.search(numeric_pattern, text)

    if match:
        try:
            return float(match.group())
        except ValueError:
            logger.warning(f"Could not convert {match.group()} to float")
            return None
    return None

def parse_reference_range(range_text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse reference range text into lower and upper bounds.

    Args:
        range_text: String containing reference range information.

    Returns:
        Tuple of (lower_bound, upper_bound) as floats. Either may be None if not found.
    """
    if not range_text:
        return None, None

    # Handle various range formats: '10-20', '< 10', '> 20', etc.
    # Pattern for ranges like '10-20', '10.5-20.5', etc.
    range_pattern = r'([-+]?\d*\.?\d+)\s*[-–]\s*([-+]?\d*\.?\d+)'
    # Pattern for values with less than/greater than: '< 10', '> 20', etc.
    lt_pattern = r'<\s*([-+]?\d*\.?\d+)'
    gt_pattern = r'>\s*([-+]?\d*\.?\d+)'

    range_match = re.search(range_pattern, range_text)
    if range_match:
        try:
            lower = float(range_match.group(1))
            upper = float(range_match.group(2))
            return lower, upper
        except ValueError:
            logger.warning(f"Could not parse range: {range_text}")

    lt_match = re.search(lt_pattern, range_text)
    if lt_match:
        try:
            upper = float(lt_match.group(1))
            return None, upper
        except ValueError:
            logger.warning(f"Could not parse less than range: {range_text}")

    gt_match = re.search(gt_pattern, range_text)
    if gt_match:
        try:
            lower = float(gt_match.group(1))
            return lower, None
        except ValueError:
            logger.warning(f"Could not parse greater than range: {range_text}")

    return None, None

def is_value_out_of_range(value: Optional[float], lower_bound: Optional[float], upper_bound: Optional[float]) -> Optional[bool]:
    """
    Determine if a value is outside the reference range.

    Args:
        value: Numeric test value.
        lower_bound: Lower reference range bound.
        upper_bound: Upper reference range bound.

    Returns:
        Boolean indicating if value is out of range, or None if cannot be determined.
    """
    if value is None:
        return None

    if lower_bound is not None and value < lower_bound:
        return True

    if upper_bound is not None and value > upper_bound:
        return True

    if lower_bound is not None and upper_bound is not None:
        return value < lower_bound or value > upper_bound

    # If we have a value but no reference range, we can't determine if it's out of range
    if lower_bound is None and upper_bound is None:
        return None

    return False

def normalize_test_name(test_name: str) -> str:
    """
    Normalize lab test names to handle variations in naming conventions.

    Args:
        test_name: Raw test name from lab report.

    Returns:
        Normalized test name.
    """
    if not test_name:
        return ""

    # Convert to lowercase
    test_name = test_name.lower()

    # Remove extra whitespace
    test_name = re.sub(r'\s+', ' ', test_name).strip()

    # Replace common abbreviations and variations
    replacements = {
        'hgb': 'hemoglobin',
        'wbc': 'white blood cell',
        'rbc': 'red blood cell',
        'hct': 'hematocrit',
        'mcv': 'mean corpuscular volume',
        'mch': 'mean corpuscular hemoglobin',
        'mchc': 'mean corpuscular hemoglobin concentration',
        'plt': 'platelet',
        'gluc': 'glucose',
        'na': 'sodium',
        'k': 'potassium',
        'cl': 'chloride',
        'co2': 'carbon dioxide',
        'bun': 'blood urea nitrogen',
        'cr': 'creatinine',
        'gfr': 'glomerular filtration rate',
        'tsh': 'thyroid stimulating hormone',
        'ft4': 'free thyroxine',
        'ft3': 'free triiodothyronine',
        'hdl': 'high-density lipoprotein',
        'ldl': 'low-density lipoprotein',
        'tg': 'triglycerides',
        'ast': 'aspartate aminotransferase',
        'alt': 'alanine aminotransferase',
        'alp': 'alkaline phosphatase',
        'tbil': 'total bilirubin',
        'dbil': 'direct bilirubin',
    }

    for abbr, full in replacements.items():
        pattern = r'\b' + abbr + r'\b'
        test_name = re.sub(pattern, full, test_name)

    return test_name

def extract_units(text: str) -> Optional[str]:
    """
    Extract measurement units from text string.

    Args:
        text: String that may contain unit information.

    Returns:
        Extracted unit string or None if not found.
    """
    if not text:
        return None

    # Common lab units
    units = [
        'g/dl', 'g/dL', 'g/L', 'g/l',
        'mg/dl', 'mg/dL', 'mg/L', 'mg/l',
        'μg/dl', 'μg/dL', 'μg/L', 'μg/l',
        'ng/dl', 'ng/dL', 'ng/L', 'ng/l',
        'pg/ml', 'pg/mL', 'pg/L', 'pg/l',
        'mmol/L', 'mmol/l', 'μmol/L', 'μmol/l',
        'mIU/L', 'mIU/l', 'IU/L', 'IU/l',
        'U/L', 'U/l', 'mU/L', 'mU/l',
        'mcg/ml', 'mcg/mL', 'mcg/L', 'mcg/l',
        '%', 'mm/hr', 'mm/h', 'sec', 'seconds',
        'fL', 'fl', 'pg', 'mEq/L', 'mEq/l',
        'cells/μL', 'cells/μl', 'cells/uL', 'cells/ul',
        'K/μL', 'K/μl', 'K/uL', 'K/ul',
        'M/μL', 'M/μl', 'M/uL', 'M/ul'
    ]

    # Build regex pattern to match any of the units
    units_pattern = '|'.join([re.escape(unit) for unit in units])
    unit_match = re.search(f'({units_pattern})', text, re.IGNORECASE)

    if unit_match:
        return unit_match.group(1)

    return None

def format_error_response(error_message: str) -> Dict[str, Any]:
    """
    Format error response for API.

    Args:
        error_message: Description of the error.

    Returns:
        Formatted error response dictionary.
    """
    return {
        "is_success": False,
        "error": error_message,
        "data": None
    }

def format_success_response(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format successful response for API.

    Args:
        data: List of extracted lab test data.

    Returns:
        Formatted success response dictionary.
    """
    return {
        "is_success": True,
        "error": None,
        "data": data
    }

def safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.

    Args:
        data: Dictionary to get value from.
        key: Key to look up.
        default: Default value to return if key not found.

    Returns:
        Value or default.
    """
    return data.get(key, default)

def validate_image_file(file_content: bytes) -> bool:
    """
    Validate if the file content is a valid image.

    Args:
        file_content: Bytes content of the file.

    Returns:
        Boolean indicating if file is a valid image.
    """
    # Check for common image file signatures (magic numbers)
    # JPEG: starts with FF D8
    # PNG: starts with 89 50 4E 47
    # TIFF: starts with 49 49 or 4D 4D
    # PDF: starts with 25 50 44 46

    if not file_content or len(file_content) < 8:
        return False

    signatures = {
        b'\xFF\xD8': 'jpeg',
        b'\x89PNG': 'png',
        b'II*\x00': 'tiff',
        b'MM\x00*': 'tiff',
        b'%PDF': 'pdf'
    }

    for sig, _ in signatures.items():
        if file_content.startswith(sig):
            return True

    return False