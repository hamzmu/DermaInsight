from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import os
import models.pretrain_models 
from models.rag_model import diagnose

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
    image_input = True
    if not file or file.filename == "":
        image_input == False

    
    
    #return templates.TemplateResponse("index.html", {"request": request})

    # Save uploaded image
    if image_input:
        save_path = f"temp_{file.filename}"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run models
        vit_results = models.pretrain_models.parallel_vit_process(save_path)
        os.remove(save_path)  # Clean up temp file


        json_rag_input = {
            "vit_output" : vit_results,
            "user_prompt" : prompt

        }
        

    else:
        json_rag_input = {
            "vit_output" : "",
            "user_prompt" : prompt

        }
    output = diagnose(json_rag_input)

    return templates.TemplateResponse("result.html", {
            "request": request,
            "result": output
        })

if __name__ == "__main__":

    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


    