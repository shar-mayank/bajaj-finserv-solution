from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any
import traceback

from app.services.image_processor import preprocess_image
from app.services.text_extractor import extract_lab_tests
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lab Report Processing API",
    description="API for extracting lab test data from medical reports",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "message": "Lab Report Processing API",
        "version": "1.0.0",
        "status": "online"
    }

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint to extract lab test data from uploaded lab report images.

    Args:
        file: The uploaded lab report image file

    Returns:
        JSON response with extracted lab test data
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed"
            )

        # Read the image file
        image_data = await file.read()

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Extract lab tests from the processed image
        lab_tests = extract_lab_tests(processed_image)

        # Prepare response
        response_data = {
            "is_success": True,
            "data": [test.to_dict() for test in lab_tests]
        }

        return JSONResponse(content=response_data)

    except HTTPException as e:
        logger.error(f"HTTP error: {str(e)}")
        return JSONResponse(
            status_code=e.status_code,
            content={"is_success": False, "error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"is_success": False, "error": f"Failed to process image: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)