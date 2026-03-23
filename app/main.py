import os
import uuid
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

DATA_ROOT = Path(".")
UPLOAD_DIR = DATA_ROOT / "uploads"
OUTPUT_DIR = DATA_ROOT / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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

    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    jobs[job_id] = {
        "status": "uploaded",
        "file": str(file_path),
        "stems": []
    }

    return {"job_id": job_id, "status": "uploaded"}

@app.post("/separate/{job_id}")
def separate(job_id: str):

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    input_file = jobs[job_id]["file"]
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(exist_ok=True)

    try:

        subprocess.run([
            "demucs",
            "-n",
            "mdx_extra_q",
            "--two-stems=vocals",
            "-o",
            str(output_dir),
            input_file
        ], check=True)

        jobs[job_id]["status"] = "completed"

        stem_dir = output_dir / "mdx_extra_q" / Path(input_file).stem

        stems = []
        if stem_dir.exists():
            for f in stem_dir.glob("*.wav"):
                stems.append(str(f))

        jobs[job_id]["stems"] = stems

        return {"job_id": job_id, "status": "completed"}

    except Exception as e:

        jobs[job_id]["status"] = "failed"
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/jobs/{job_id}")
def job_status(job_id: str):

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]

@app.get("/download/{job_id}/{stem}")
def download(job_id: str, stem: str):

    path = OUTPUT_DIR / job_id / "mdx_extra_q"

    for root, dirs, files in os.walk(path):
        if stem in files:
            return FileResponse(os.path.join(root, stem))

    raise HTTPException(status_code=404, detail="Stem not found")
