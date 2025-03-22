from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn





app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <title>AI Prompt + Image Upload</title>
            <link rel="stylesheet" type="text/css" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h1>DermaInsight</h1>
                <form action="/upload/" enctype="multipart/form-data" method="post">
                    <input type="text" name="prompt" placeholder="Enter your prompt here"/><br/>
                    <input name="file" type="file" accept=".jpg, .jpeg, .png"/><br/>
                    <button type="submit">Submit</button>
                </form>
            </div>
        </body>
    </html>
    """
    return content

@app.post("/upload/")
async def upload(prompt: str = Form(...), file: UploadFile = File(...)):
    # Process prompt and image here
    return {"message": "Received!", "prompt": prompt, "filename": file.filename}

# Run app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
