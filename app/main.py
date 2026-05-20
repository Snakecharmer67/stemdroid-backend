import os
import uuid
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from app.video_effects import render_match_morph_transition, mux_audio_from_clip

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

    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    jobs[job_id] = {
        "status": "uploaded",
        "file": str(file_path),
        "stems": [],
        "outputs": []
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

@app.post("/effects/match-morph")
async def match_morph_transition(
    clip_a: UploadFile = File(...),
    clip_b: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    clip_a_path = UPLOAD_DIR / f"{job_id}_a_{Path(clip_a.filename).name}"
    clip_b_path = UPLOAD_DIR / f"{job_id}_b_{Path(clip_b.filename).name}"

    with open(clip_a_path, "wb") as buffer:
        buffer.write(await clip_a.read())

    with open(clip_b_path, "wb") as buffer:
        buffer.write(await clip_b.read())

    jobs[job_id] = {
        "status": "processing",
        "effect": "match_morph_no_blur",
        "inputs": [str(clip_a_path), str(clip_b_path)],
        "outputs": []
    }

    try:
        rendered_path = render_match_morph_transition(
            clip_a=clip_a_path,
            clip_b=clip_b_path,
            output_dir=job_dir,
            slowmo_seconds=0.8,
            morph_seconds=0.33,
            restore_seconds=0.5,
            no_blur=True,
        )

        final_path = mux_audio_from_clip(rendered_path, clip_a_path, job_dir)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["outputs"] = [str(final_path)]

        return {
            "job_id": job_id,
            "status": "completed",
            "effect": "match_morph_no_blur",
            "download": f"/download-output/{job_id}/{final_path.name}"
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return JSONResponse(
            status_code=500,
            content={"job_id": job_id, "status": "failed", "error": str(e)}
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

@app.get("/download-output/{job_id}/{filename}")
def download_output(job_id: str, filename: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    for output in jobs[job_id].get("outputs", []):
        path = Path(output)
        if path.name == filename and path.exists():
            return FileResponse(path, filename=path.name, media_type="video/mp4")

    raise HTTPException(status_code=404, detail="Output not found")
