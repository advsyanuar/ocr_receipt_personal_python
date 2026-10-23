import base64
import time
import uuid
from pathlib import Path

import anyio
import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

SHARED_DIR = Path("/home/manonthemoon/Projects/OCR-SHARED")
UPLOADS_DIR = SHARED_DIR / "Uploads"
PARSED_DIR = SHARED_DIR / "Parsed"

DOTNET_ENDPOINT = "http://localhost:5218/api/v1/ocr"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

app = FastAPI(title="OCR Service")
origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llm = ChatOllama(model="qwen2.5vl:3b", temperature=0.0)

tasks: dict[str, dict] = {}


class ProcessImageRequest(BaseModel):
    filename: str = Field(
        ...,
        description="Filename of the image stored in OCR-SHARED/Uploads",
        examples=["9613318a-62cd-4401-9a67-eca48f858008.jpeg"],
    )


def encode_image(img_path: Path | str) -> str:
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


async def run_ocr_task(task_id: str, file_path: Path):
    try:
        start_time = time.perf_counter()
        base64_image = encode_image(file_path)

        ext = file_path.suffix.lower().lstrip(".")
        mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Perform OCR on this image. Extract all the text available.",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{base64_image}",
                },
            ]
        )

        response = await llm.ainvoke([message])
        end_time = time.perf_counter()
        elapsed_time = round(end_time - start_time, 4)

        output_filename = f"{file_path.stem}.txt"
        output_path = PARSED_DIR / output_filename

        await anyio.Path(output_path).write_text(str(response.content), encoding="utf-8")

        payload = {
            "rawTextPath": str(output_path),
            "isProcessed": True,
        }

        async with httpx.AsyncClient() as client:
            get_response = await client.get(DOTNET_ENDPOINT)
            get_response.raise_for_status()
            records = get_response.json()

            target_str = str(file_path)
            matched_record = next(
                (
                    item for item in records
                    if item.get("imagePath") == target_str
                    or Path(item.get("imagePath", "")).resolve() == file_path.resolve()
                    or Path(item.get("imagePath", "")).name == file_path.name
                ),
                None,
            )

            if not matched_record:
                raise ValueError(
                    f"No matching .NET record found with imagePath matching '{file_path.name}'"
                )

            image_id = matched_record["id"]

            patch_response = await client.patch(
                f"{DOTNET_ENDPOINT}/{image_id}",
                json=payload,
            )
            patch_response.raise_for_status()

        tasks[task_id] = {
            "processed": True,
            "file": output_filename,
            "timeElapsed": elapsed_time,
            "status": "completed",
        }
    except Exception as e:  # noqa: BLE001
        tasks[task_id] = {
            "processed": False,
            "file": None,
            "timeElapsed": None,
            "status": "failed",
            "error": str(e),
        }


@app.post("/process-image")
async def process_image(
    request: ProcessImageRequest,
    background_tasks: BackgroundTasks,
):
    target_file = UPLOADS_DIR / request.filename
    if not target_file.is_file():
        fallback_file = SHARED_DIR / request.filename
        if fallback_file.is_file():
            target_file = fallback_file
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Image file '{request.filename}' not found in {UPLOADS_DIR}",
            )

    ext = target_file.suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTS))}",
        )

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "processed": False,
        "file": None,
        "timeElapsed": None,
        "status": "pending",
    }

    background_tasks.add_task(run_ocr_task, task_id, target_file)

    return {
        "message": "file processed and status: pending",
        "task_id": task_id,
        "status": "pending",
    }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5222, reload=True)
