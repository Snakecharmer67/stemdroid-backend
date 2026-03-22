import os
import sys
import uuid
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "."))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", DATA_ROOT / "uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DATA_ROOT / "outputs"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}

@app.get("/")
def root():
    return {"status": "stem separation server running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    original_name = Path(file.filename).name
    file_path = UPLOAD_DIR / f"{job_id}_{original_name}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    jobs[job_id] = {
        "job_id": job_id,
        "status": "uploaded",
        "message": "File uploaded",
        "file": str(file_path),
        "stems": []
    }

    return {"job_id": job_id, "status": "uploaded"}

@app.post("/separate/{job_id}")
def separate(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "job not found"})

    input_file = Path(jobs[job_id]["file"])
    output_folder = OUTPUT_DIR / job_id
    output_folder.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-o",
        str(output_folder),
        str(input_file)
    ]

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Separation in progress"
        jobs[job_id]["stems"] = []

        subprocess.run(command, check=True)

        track_stem_dir = output_folder / "htdemucs" / input_file.stem

        stem_paths = []
        if track_stem_dir.exists():
            for stem_file in sorted(track_stem_dir.glob("*.wav")):
                stem_paths.append(str(stem_file))

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Separation completed"
        jobs[job_id]["stems"] = stem_paths

        return {
            "job_id": job_id,
            "status": "completed",
            "output": str(output_folder)
        }

    except subprocess.CalledProcessError as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Separation failed: {e}"
        jobs[job_id]["stems"] = []

        return {
            "job_id": job_id,
            "status": "failed"
        }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return jobs[job_id]

@app.get("/download/{job_id}/{stem_file_name}")
def download_stem(job_id: str, stem_file_name: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job not found")

    stem_paths = jobs[job_id].get("stems", [])
    if not stem_paths:
        raise HTTPException(status_code=404, detail="no stems available for this job")

    for stem_path in stem_paths:
        path_obj = Path(stem_path)
        if path_obj.name == stem_file_name and path_obj.exists():
            return FileResponse(
                path=str(path_obj),
                filename=path_obj.name,
                media_type="audio/wav"
            )

    raise HTTPException(status_code=404, detail="stem file not found")
