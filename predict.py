import json
import os
import random
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from cog import BaseModel, BasePredictor, Input, Path
import requests
import websocket

COMFY_HOST = "127.0.0.1:8188"
COMFY_PYTHON = "/root/comfy_env/bin/python"


DEFAULT_STYLE = (
    "late-night smooth jazz radio station bumper, smoky tenor saxophone, warm rhodes electric piano chords, brush snare, 75 bpm, deep resonant male vocals"
)

DEFAULT_LYRICS = """[verse]
When city shadows turn to blue,
We play the midnight sound for you.
[chorus]
Slip into velvet, ease your mind,
The smoothest rhythm you can find."""

class Output(BaseModel):
  audio: Path
  score_abc: str


class Predictor(BasePredictor):

  def setup(self):
    """Starts ComfyUI headless server in the background using its dedicated virtualenv."""
    print("Starting background ComfyUI instance in isolated virtualenv...")
    cmd = [
        COMFY_PYTHON,
        "/root/ComfyUI/main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--fast",
        "fp16_accumulation",
        "--verbose",
        "INFO",
    ]
    self.comfy_process = subprocess.Popen(cmd)

    # Poll port until ComfyUI is online
    ready = False
    for _ in range(60):
      if self.comfy_process.poll() is not None:
        raise RuntimeError(
            f"ComfyUI process exited prematurely with code"
            f" {self.comfy_process.returncode}"
        )

      try:
        res = requests.get(f"http://{COMFY_HOST}/system_stats", timeout=1)
        if res.status_code == 200:
          ready = True
          break
      except Exception:
        time.sleep(1)

    if not ready:
      raise RuntimeError("ComfyUI failed to start within 60 seconds.")
    print("ComfyUI server is online and ready for jobs.")

  def predict(
      self,
      style: str = Input(
          description="Genre, instruments, mood, tempo, vocal character",
          default=DEFAULT_STYLE,
      ),
      lyrics: str = Input(
          description=(
              "Song lyrics or bracketed musical structure tags ([intro], [solo],"
              " etc.)"
          ),
          default=DEFAULT_LYRICS,
      ),
      audio_format: str = Input(
          description="Audio output format",
          choices=["mp3", "wav", "flac"],
          default="mp3",
      ),
      cot: str = Input(
          description=(
              "Planning mode: 'full' (melody + chords), 'melody' (melody only),"
              " 'off' (direct synthesis)"
          ),
          choices=["full", "melody", "off"],
          default="full",
      ),
      max_duration: float = Input(
          description=(
              "Target song duration in seconds (controls latent audio frames)"
          ),
          ge=15.0,
          le=360.0,
          default=60.0,
      ),
      steps: int = Input(
          description=(
              "Acoustic diffusion steps (15-20 fast, 25-32 standard, 50 max)"
          ),
          ge=10,
          le=100,
          default=25,
      ),
      sampler_name: str = Input(
          description="Diffusion ODE solver algorithm",
          choices=["dpm_2", "euler", "dpmpp_2m"],
          default="dpm_2",
      ),
      scheduler: str = Input(
          description="Noise reduction schedule curve",
          choices=["sgm_uniform", "karras", "simple"],
          default="sgm_uniform",
      ),
      max_abc_tokens: int = Input(
          description=(
              "Maximum tokens generated during the ABC musical score planning"
              " stage"
          ),
          ge=256,
          le=8192,
          default=4096,
      ),
      abc_temperature: float = Input(
          description=(
              "Randomness of score composition (lower = predictable pop, higher"
              " = complex/jazz)"
          ),
          ge=0.1,
          le=2.0,
          default=0.70,
      ),
      custom_abc: str = Input(
          description=(
              "(Optional) Supply your own ABC score to bypass the score"
              " planning step"
          ),
          default=None,
      ),
      seed: int = Input(
          description="Random seed for reproducibility (-1 for random)",
          default=-1,
      ),
  ) -> Output:
    """Modifies the node graph and executes ComfyUI headless."""
    if seed < 0:
      seed = random.randint(0, 2**32 - 1)
    print(f"Executing ComfyUI job with seed: {seed}")

    formatted_lyrics = lyrics.replace("\\n", "\n").strip()

    # 1. Load exported workflow_api.json
    with open("workflow_api.json", "r", encoding="utf-8") as f:
      prompt = json.load(f)

    # 2. Inject inputs by matching class_type
    for node_id, node in prompt.items():
      class_type = node.get("class_type")

      # Stage 1: ABC Score Generation
      if class_type == "YuE2GenerateABC":
        node["inputs"]["clip"] = style
        node["inputs"]["max_abc_tokens"] = max_abc_tokens
        node["inputs"]["temperature"] = abc_temperature
        node["inputs"]["seed"] = seed
        node["inputs"]["mode"] = cot
        if custom_abc and custom_abc.strip():
          node["inputs"]["custom_abc"] = custom_abc.strip()

      # Stage 2: Music Conditioning Engine
      elif class_type == "YuE2GenerateMusic":
        node["inputs"]["max_duration"] = max_duration
        node["inputs"]["seed"] = seed
        node["inputs"]["mode"] = cot

      # Stage 3: Diffusion KSampler
      elif class_type == "KSampler":
        node["inputs"]["steps"] = steps
        node["inputs"]["sampler_name"] = sampler_name
        node["inputs"]["scheduler"] = scheduler
        node["inputs"]["seed"] = seed
        node["inputs"]["cfg"] = 1.0

    # 3. Submit workflow to ComfyUI
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    response = json.loads(urllib.request.urlopen(req).read())
    prompt_id = response["prompt_id"]

    # 4. Wait for execution to finish via WebSocket
    while True:
      out = ws.recv()
      if isinstance(out, str):
        message = json.loads(out)
        if message["type"] == "executing":
          data = message["data"]
          if data["node"] is None and data["prompt_id"] == prompt_id:
            break
      else:
        continue
    ws.close()

    # 5. Locate generated output audio
    output_dir = "/root/ComfyUI/output"
    audio_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith((".flac", ".wav", ".mp3"))
    ]
    if not audio_files:
      raise RuntimeError("No audio file found in ComfyUI output directory.")

    latest_audio = max(audio_files, key=os.path.getctime)

    # 6. Extract generated score.abc if present
    abc_text = ""
    txt_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith((".abc", ".txt"))
    ]
    if txt_files:
      latest_abc = max(txt_files, key=os.path.getctime)
      with open(latest_abc, "r", encoding="utf-8") as f:
        abc_text = f.read()

    # 7. Transcode to requested format
    final_output = latest_audio
    if audio_format == "mp3" and not latest_audio.endswith(".mp3"):
      final_output = os.path.splitext(latest_audio)[0] + "_out.mp3"
      subprocess.run(
          ["ffmpeg", "-y", "-i", latest_audio, "-b:a", "320k", final_output],
          check=True,
      )
    elif audio_format == "wav" and not latest_audio.endswith(".wav"):
      final_output = os.path.splitext(latest_audio)[0] + "_out.wav"
      subprocess.run(
          ["ffmpeg", "-y", "-i", latest_audio, final_output], check=True
      )

    return Output(audio=Path(final_output), score_abc=abc_text)