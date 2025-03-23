from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import os
import models.pretrain_models 

app = FastAPI()

# Serve static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload(request: Request, prompt: str = Form(""), file: UploadFile = File(None)):
    # Check if user uploaded an image
    if not file or file.filename == "":
        # Simply return the original form again
        return templates.TemplateResponse("index.html", {"request": request})

    # Save uploaded image
    save_path = f"temp_{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run models
    vit_results = models.pretrain_models.parallel_vit_process(save_path)
    os.remove(save_path)  # Clean up temp file
    user_prompt = prompt

    #TODO: f(vit_results, user_prompt) 

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prompt": prompt,
        "result": vit_results
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
