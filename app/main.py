from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import re
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lab Report Parser API",
    description="Extracts lab test data from images using OCR.",
    version="0.1.0"
)


class LabTest:
    def __init__(self, test_name: str, test_value: str, bio_reference_range: str = "", test_unit: str = "", lab_test_out_of_range: bool = False):
        self.test_name = test_name
        self.test_value = test_value
        self.bio_reference_range = bio_reference_range
        self.test_unit = test_unit
        self.lab_test_out_of_range = lab_test_out_of_range

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "test_value": self.test_value,
            "bio_reference_range": self.bio_reference_range,
            "test_unit": self.test_unit,
            "lab_test_out_of_range": self.lab_test_out_of_range
        }

def preprocess_image(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opening

def is_value_out_of_range(value: str, reference_range: str) -> bool:
    try:
        if "-" in reference_range:
            low, high = reference_range.split("-")
            low = float(low.strip())
            high = float(high.strip())
            test_value = float(value.strip())
            return test_value < low or test_value > high
        else:
            return False
    except (ValueError, TypeError):
        return False

def extract_json_data(text: str) -> List[LabTest]:
    try:
        json_pattern = r'\{\s*"is_success":\s*true,\s*"data":\s*\[.*?\]\s*\}'
        json_match = re.search(json_pattern, text, re.DOTALL)

        if json_match:
            json_text = json_match.group(0)
            json_text = re.sub(r'\s+', ' ', json_text)
            json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)
            json_text = json_text.replace("'", '"')

            data = json.loads(json_text)
            lab_tests = []

            for test_data in data.get("data", []):
                test = LabTest(
                    test_name=test_data.get("test_name", ""),
                    test_value=test_data.get("test_value", ""),
                    bio_reference_range=test_data.get("bio_reference_range", ""),
                    test_unit=test_data.get("test_unit", ""),
                    lab_test_out_of_range=test_data.get("lab_test_out_of_range", False)
                )
                lab_tests.append(test)

            return lab_tests
    except Exception as e:
        logger.error(f"Error extracting JSON data: {e}")

    return []

def extract_tabular_data(text: str) -> List[LabTest]:
    """Extract data from tabular format like in images 2, 4, and 5."""
    lab_tests = []

    patterns = [
        (
            r"(?P<test_name>[A-Za-z\s\(\)]+)\s*:\s*(?P<test_value>[\d\.]+)\s*(?P<test_unit>[A-Za-z%/]+)?\s*(?P<bio_reference_range>[\d\.\-]+)",
            lambda m: LabTest(
                test_name=m.group("test_name").strip(),
                test_value=m.group("test_value").strip(),
                test_unit=m.group("test_unit").strip() if m.group("test_unit") else "",
                bio_reference_range=m.group("bio_reference_range").strip(),
                lab_test_out_of_range=is_value_out_of_range(m.group("test_value"), m.group("bio_reference_range"))
            )
        ),
        (
            r"(?P<test_name>[A-Za-z\s\(\)]+)\s+(?P<test_value>[\d\.]+)\s+(?P<test_unit>[A-Za-z%/]+)?\s+(?P<bio_reference_range>[\d\.\-]+)",
            lambda m: LabTest(
                test_name=m.group("test_name").strip(),
                test_value=m.group("test_value").strip(),
                test_unit=m.group("test_unit").strip() if m.group("test_unit") else "",
                bio_reference_range=m.group("bio_reference_range").strip(),
                lab_test_out_of_range=is_value_out_of_range(m.group("test_value"), m.group("bio_reference_range"))
            )
        )
    ]

    specific_tests = [
        r"C-Reactive Protein \(CRP\)\s*:\s*(?P<test_value>[\d\.]+)\s*\[H\]\s*(?P<test_unit>mg/L)\s*Biological\s*Ref\.\s*Range\s*(?P<bio_reference_range>[\d\-]+)\s*mg/L",

        r"HEMOGLOBIN\s*(?P<test_value>[\d\.]+)\s*gm%\s*(?P<bio_reference_range>[\d\.\-]+)\s*gm%",

        r"Haemoglobin\s*:\s*(?P<test_value>[\d\.]+)\s*\[L\]\s*(?P<test_unit>gm/dl)\s*(?P<bio_reference_range>[\d\.\-]+)",
        r"Platelet Count\s*:\s*(?P<test_value>[\d\.]+)\s*/uL\s*(?P<bio_reference_range>[\d\.\-]+)",
    ]

    for pattern, create_test in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            lab_tests.append(create_test(match))

    for pattern in specific_tests:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                test_value = match.group("test_value")
                bio_range = match.group("bio_reference_range")
                test_unit = match.group("test_unit") if "test_unit" in match.groupdict() else ""
                test_name = match.group(0).split(":")[0].strip() if ":" in match.group(0) else match.group(0).split()[0].strip()

                lab_tests.append(LabTest(
                    test_name=test_name,
                    test_value=test_value,
                    bio_reference_range=bio_range,
                    test_unit=test_unit,
                    lab_test_out_of_range=is_value_out_of_range(test_value, bio_range)
                ))
            except (IndexError, AttributeError) as e:
                logger.error(f"Error processing specific test pattern: {e}")

    crp_match = re.search(r"C-Reactive Protein \(CRP\).*?(?P<test_value>[\d\.]+)\s*\[H\]", text)
    if crp_match:
        crp_range_match = re.search(r"Biological\s*Ref\.\s*Range\s*(?P<bio_range>[\d\-]+)\s*mg/L", text)
        bio_range = crp_range_match.group("bio_range") if crp_range_match else "0-5"

        lab_tests.append(LabTest(
            test_name="C-Reactive Protein (CRP)",
            test_value=crp_match.group("test_value"),
            bio_reference_range=bio_range,
            test_unit="mg/L",
            lab_test_out_of_range=True if "[H]" in crp_match.group(0) else is_value_out_of_range(crp_match.group("test_value"), bio_range)
        ))

    return lab_tests

def extract_lab_tests(image_data) -> List[LabTest]:
    """Extract lab tests from the image data."""
    processed_img = preprocess_image(image_data)

    ocr_text = pytesseract.image_to_string(processed_img)
    logger.info(f"OCR Text: {ocr_text[:200]}...")  # Log first 200 chars for debugging

    lab_tests = extract_json_data(ocr_text)
    if not lab_tests:
        lab_tests = extract_tabular_data(ocr_text)

    if "CRP REPORT" in ocr_text:
        crp_value_match = re.search(r"(?:Result|:)\s*(?P<value>[\d\.]+)", ocr_text)
        crp_range_match = re.search(r"Biological\s*Ref\.\s*Range\s*(?P<range>[\d\-]+)", ocr_text)

        if crp_value_match:
            value = crp_value_match.group("value")
            bio_range = crp_range_match.group("range") if crp_range_match else "0-5"

            lab_tests.append(LabTest(
                test_name="C-Reactive Protein (CRP)",
                test_value=value,
                bio_reference_range=bio_range,
                test_unit="mg/L",
                lab_test_out_of_range=is_value_out_of_range(value, bio_range)
            ))

    if "COMPLETE BLOOD COUNT" in ocr_text or "CBC" in ocr_text:
        parameters = [
            ("Hemoglobin", r"(?:Haemoglobin|Hemoglobin).*?(?P<value>[\d\.]+).*?(?P<range>[\d\.\-]+)"),
            ("RBC Count", r"(?:R\.B\.C\.|RBC).*?Count.*?(?P<value>[\d\.]+).*?(?P<range>[\d\.\-]+)"),
            ("WBC Count", r"(?:W\.B\.C\.|WBC).*?Count.*?(?P<value>[\d\.]+).*?(?P<range>[\d\.\-]+)"),
            ("Platelet Count", r"Platelet\s*Count.*?(?P<value>[\d\.]+).*?(?P<range>[\d\.\-]+)"),
            ("PCV", r"(?:Packed\s*Cell\s*Volume|PCV|HCT).*?(?P<value>[\d\.]+).*?(?P<range>[\d\.\-]+)")
        ]

        for name, pattern in parameters:
            match = re.search(pattern, ocr_text)
            if match:
                lab_tests.append(LabTest(
                    test_name=name,
                    test_value=match.group("value"),
                    bio_reference_range=match.group("range"),
                    test_unit="",
                    lab_test_out_of_range=is_value_out_of_range(match.group("value"), match.group("range"))
                ))

    return lab_tests
