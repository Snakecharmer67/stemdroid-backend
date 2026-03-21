from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"

for folder in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="StemDroid API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "StemDroid backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed = {".mp3", ".wav", ".flac", ".m4a"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    job_id = str(uuid.uuid4())
    saved_name = f"{job_id}{suffix}"
    saved_path = UPLOAD_DIR / saved_name

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "job_id": job_id,
        "original_filename": file.filename,
        "saved_filename": saved_name,
        "saved_path": str(saved_path),
        "next_step": f"/separate/{job_id}"
    }

@app.post("/separate/{job_id}")
def separate_stub(job_id: str):
    fake_output_dir = OUTPUT_DIR / job_id
    fake_output_dir.mkdir(parents=True, exist_ok=True)

    stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    for stem in stems:
        stem_path = fake_output_dir / stem
        if not stem_path.exists():
            stem_path.write_bytes(b"")

    return {
        "job_id": job_id,
        "status": "completed_stub",
        "message": "Stub separation complete. Real model hookup comes next.",
        "stems": [f"/download/{job_id}/{stem}" for stem in stems]
    }

@app.get("/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str):
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return JSONResponse(status_code=404, content={"detail": "Job not found"})

    files = sorted([p.name for p in job_dir.iterdir() if p.is_file()])
    return {"job_id": job_id, "files": files}
