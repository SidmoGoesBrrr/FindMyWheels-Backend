from fastapi import FastAPI, File, Form, UploadFile,BackgroundTasks, Body, Depends, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
import base64
import io
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import logging
import typing
import google.generativeai as genai
from typing import Dict
import os
import json
from datetime import datetime
logging.basicConfig(filename='api.log', encoding='utf-8', level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.debug('Starting application')
logger.info('Logger is configured and working')

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY") #get the google api key from the environment variables
from PIL import Image
genai.configure(api_key=GOOGLE_API_KEY)

cred = credentials.Certificate(r"ai-stuff-vehicles-firebase-adminsdk-2x41p-45b8355de9.json") #store this locally

default_app=firebase_admin.initialize_app(cred,
                                          {
    'databaseURL': 'https://ai-stuff-vehicles-default-rtdb.firebaseio.com/'
})

API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")


app = FastAPI()
def extract_parking_lots(db_data, parent_path=""):
    """Extract parking lot paths from Firebase Realtime Database data."""
    parking_lots = []
    for key, value in db_data.items():
        if isinstance(value, dict):
            # Recursively call extract_parking_lots for nested data
            nested_lots = extract_parking_lots(value, f"{parent_path}/{key}")
            parking_lots.extend(nested_lots)
        else:
            # Split the parent path by '/' and take the first two directories
            directories = parent_path.split('/')[1:3]
            parking_lot_path = '/'.join(directories)
            if parking_lot_path not in parking_lots:
                parking_lots.append(parking_lot_path)
    return list(set(parking_lots))


tasks: Dict[str, dict] = {}

def save_results(slot, results, image_path, fb_path):
    """Save results to JSON file and Firebase, ensuring JSON format is correct."""
    try:
            details = json.loads(results) if isinstance(results, str) else results
            details.update({"parking_slot": slot, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            with open(f'results_slot{slot}.json', 'w') as f:
                json.dump(details, f, indent=4)
            logger.info(f"Saved results to {fb_path}")
            db.reference(fb_path).child(f'slot_{slot}').set(details)
    except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from the results: {e}")

async def process_image(base64_image, slot, file_location, fb_path):
    """Process an image, get info from Gemini and save the results to a JSON file and Firebase."""
    try:
        # Decode base64 image string to bytes
        image_bytes = decode_image(base64_image)
        img = Image.open(io.BytesIO(image_bytes))
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(["""
    I will be submitting a picture of a car, front facing. It will have an indian numberplate and will have a color and a logo. Please return a json in the following format
        {
        \"car_brand\": \"Tata\",
        \"car_model\" : \"Nexon\",
        \"color\" : \"Red\",
        \"license_plate\": \"MH12AB1234\"
        }
        Please change the car information based on the image uploaded. Also, if you are unable to identify something or everything type \"N/A\" for that particular image.
        Bharat's vehicle number plates follow a format: XX YY ZZZZ, where XX represents the state code, YY the district RTO code, and ZZZZ a unique vehicle ID. For example, MH 02 AB 1234 indicates Maharashtra (MH), Mumbai Central RTO (02), with a unique vehicle ID (AB 1234). Please ensure the license plate is in this format. If the license plate is not visible, please type \"N/A\".
        
    """, img])
        response.resolve()
        save_results(slot, response.text, file_location, fb_path)
    except Exception as e:
        logger.error("Error processing image:", e)

def decode_image(data_url: str) -> bytes:
    """ Decode a base64 image from a data URL. """
    if data_url.startswith('data:image'):
        # Split on comma to remove data URL scheme
        base64_str = data_url.split(",")[1]
    else:
        raise ValueError("Invalid data URL provided")
    return base64.b64decode(base64_str)

@app.post("/image", dependencies=[Depends(get_api_key)])
async def receive_image(background_tasks: BackgroundTasks, image: str = Body(...), path: str = Body(...), slot: int = Body(...)):
    """Receive an image and process it in the background."""
    logger.info(f"Received image for slot {slot}")
    logger.info(f"Path: {path}")

    try:
        # Decode base64 image string to bytes
        image_bytes = base64.b64decode(image)
        
        # Save the decoded image to a file
        file_location = f"images/slot{slot}.jpg"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as file:
            file.write(image_bytes)

        # Add a background task to process the image
        background_tasks.add_task(process_image, image, slot, file_location, path)  # Make sure to pass `image`, not `image_bytes`
        logger.info(f"Image saved to {file_location}. Running AI model to get information...")
        return {"message": f"Image saved to {file_location}. Running AI model to get information..."}
    except Exception as e:
        logger.error("Oops something went wrong:", e)
        return {"error": str(e)}


@app.get("/getdb", dependencies=[Depends(get_api_key)])
async def getdb(path: typing.Optional[str] = None):
    """Get parking lot data from Firebase Realtime Database."""
    if not path:
        path = "parking"  # Assume this is the root path for padarking data
    try:
        ref = db.reference(path)
        db_data = ref.get()
        if not db_data:
            raise HTTPException(status_code=404, detail="Document not found")

        # Extract and organize parking lot information
        parking_lots = extract_parking_lots(db_data)
        return {"parkings": parking_lots}
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while retrieving the data")
    
