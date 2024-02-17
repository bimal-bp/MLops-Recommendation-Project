from fastapi import FastAPI, Form, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
from pydantic import BaseModel
from src.pipeline.prediction_pipeline import CustomData, predictpipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")


class PredictionRequest(BaseModel):
    user_id: str
    product_id: str


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predictdata", response_class=HTMLResponse)
def predict_datapoint(request: Request, prediction_request: PredictionRequest = Depends()):
    data = CustomData(
        user_id=prediction_request.user_id,
        product_id=prediction_request.product_id
    )
    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return templates.TemplateResponse("home.html", {"request": request, "results": results[0][0]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
