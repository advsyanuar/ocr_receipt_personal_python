import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

@app.post("/ocr")
async def post(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    original_ext = Path(image.filename or "").suffix.lower()
    ext = original_ext if original_ext in ALLOWED_EXTS else ".jpg"

    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename

    contents = await image.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "message": "File uploaded successfully",
        "file_path": str(file_path)
    }
