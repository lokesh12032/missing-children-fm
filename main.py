from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import aiofiles
from pathlib import Path
from uuid import uuid4
import asyncio
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    missing: UploadFile = File(...),
    found: UploadFile = File(...)
):
    upload_dir = BASE_DIR / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    m_name = f"{uuid4().hex}{Path(missing.filename).suffix}"
    f_name = f"{uuid4().hex}{Path(found.filename).suffix}"

    m_path = upload_dir / m_name
    f_path = upload_dir / f_name

    async with aiofiles.open(m_path, "wb") as m:
        await m.write(await missing.read())

    async with aiofiles.open(f_path, "wb") as f:
        await f.write(await found.read())

    try:
        result = await asyncio.to_thread(
            DeepFace.verify,
            str(m_path),
            str(f_path),
            model_name="Facenet",
            detector_backend="mtcnn"
        )

        match = "YES" if result["verified"] else "NO"
        confidence = round((1 - result["distance"]) * 100, 2)

    except:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": "Face not detected"}
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "match": match,
            "confidence": confidence,
            "missing_img": f"/static/uploads/{m_name}",
            "found_img": f"/static/uploads/{f_name}"
        }
    )
