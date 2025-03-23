from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import os
import logging
import models.pretrain_models
import requests

app = FastAPI()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Serve static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload(request: Request, prompt: str = Form(""), file: UploadFile = File(None)):
    try:
        vit_results = {}
        image_input = bool(file and file.filename)
        
        if image_input:
            try:
                save_path = f"temp_{file.filename}"
                with open(save_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                vit_results = models.pretrain_models.parallel_vit_process(save_path)
                os.remove(save_path)
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                image_input = False

        rag_input = {
            "vit_analysis": vit_results,
            "user_input": prompt,
            "combined_query": f"Image analysis: {vit_results}. User description: {prompt}"
        }
        
        try:
            # Call the RAG API
            response = requests.post(
                "http://127.0.0.1:8001/diagnose/",
                json=rag_input
            )
            response.raise_for_status()
            output = response.json().get("diagnosis", "No diagnosis available.")
        except Exception as e:
            logger.error(f"Diagnosis API call failed: {e}")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": "Diagnosis engine unavailable. Please try again."
            })
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_analysis": vit_results,
            "user_input": prompt,
            "diagnosis": output,
            "show_image": image_input
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "An unexpected error occurred. Please try again."
        })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)