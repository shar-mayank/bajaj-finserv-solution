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

app = FastAPI(title="Lab Report Processing API")

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
