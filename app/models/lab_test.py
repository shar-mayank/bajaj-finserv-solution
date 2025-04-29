"""
Lab Test model for the lab report processing application.

This module defines the LabTest class which represents
a single laboratory test result with its associated data.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class LabTest(BaseModel):
    """
    Represents a laboratory test result.

    Attributes:
        test_name: The name of the laboratory test
        test_value: The measured value of the test
        bio_reference_range: The normal reference range for the test
        test_unit: The unit of measurement for the test value
        lab_test_out_of_range: Boolean flag indicating if the test value is outside the reference range
    """
    test_name: str
    test_value: str
    bio_reference_range: str = ""
    test_unit: str = ""
    lab_test_out_of_range: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the LabTest object to a dictionary.

        Returns:
            A dictionary representation of the LabTest object
        """
        return {
            "test_name": self.test_name,
            "test_value": self.test_value,
            "bio_reference_range": self.bio_reference_range,
            "test_unit": self.test_unit,
            "lab_test_out_of_range": self.lab_test_out_of_range
        }

    @classmethod
    def create(cls,
               test_name: str,
               test_value: str,
               bio_reference_range: str = "",
               test_unit: str = "",
               lab_test_out_of_range: Optional[bool] = None) -> "LabTest":
        """
        Factory method to create a LabTest object with automatic range checking.

        Args:
            test_name: The name of the laboratory test
            test_value: The measured value of the test
            bio_reference_range: The normal reference range for the test
            test_unit: The unit of measurement for the test value
            lab_test_out_of_range: Boolean flag indicating if the test value is outside the reference range
                                  (if None, it will be automatically calculated)

        Returns:
            A new LabTest object
        """
        # If lab_test_out_of_range is not provided, calculate it
        if lab_test_out_of_range is None and bio_reference_range:
            lab_test_out_of_range = cls.is_value_out_of_range(test_value, bio_reference_range)

        return cls(
            test_name=test_name,
            test_value=test_value,
            bio_reference_range=bio_reference_range,
            test_unit=test_unit,
            lab_test_out_of_range=lab_test_out_of_range
        )

    @staticmethod
    def is_value_out_of_range(value: str, reference_range: str) -> bool:
        """
        Determine if a test value is outside the reference range.

        Args:
            value: The test value as a string
            reference_range: The reference range as a string (typically in format 'min-max')

        Returns:
            True if the value is outside the reference range, False otherwise
        """
        try:
            # Handle cases where the reference range is a simple range like "12.0-15.0"
            if "-" in reference_range:
                low, high = reference_range.split("-")
                low = float(low.strip())
                high = float(high.strip())
                test_value = float(value.strip())
                return test_value < low or test_value > high
            # Handle special cases like "<5" or ">10"
            elif "<" in reference_range:
                limit = float(reference_range.replace("<", "").strip())
                test_value = float(value.strip())
                return test_value >= limit
            elif ">" in reference_range:
                limit = float(reference_range.replace(">", "").strip())
                test_value = float(value.strip())
                return test_value <= limit
            # Handle cases with text or other formats
            else:
                return False
        except (ValueError, TypeError):
            # If we can't parse the values, return False by default
            return False