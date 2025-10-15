from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.inference import predict_image
import time

app = FastAPI(title="Cloud Detection API")

@app.get("/")
def root():
    return {"message": "Cloud Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()

    start_time = time.time()          # 🔹 start timer
    detections = predict_image(file_bytes)
    elapsed = round(time.time() - start_time, 2)  # in seconds

    return JSONResponse(
        content={"detections": detections, "inference_time_sec": elapsed}
    )