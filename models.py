from pydantic import BaseModel

class PredictionResponse(BaseModel):
    detections: list