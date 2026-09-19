import os
import random
import subprocess
import tempfile
from cog import BaseModel, BasePredictor, Input, Path
from yue2 import YuE2Pipeline

DEFAULT_STYLE = (
    "french acid jazz, french house, nu jazz, deep jazz house groove, "
    "detroit underground house music, 118 bpm, punchy electronic four-on-the-floor kick, "
    "warm fender rhodes, filtered disco house chords, deep analog pumping bassline, "
    "atmospheric jazz flute solos, soulful expressive female vocal loops, late night parisian lounge, vinyl warmth"
)

DEFAULT_LYRICS = """[intro - low-pass filtered rhodes chords slowly opening up, 118 bpm electronic house kick enters with crisp closed hi-hats]

[verse - soulful vocal]
Feel the rhythm lift you up...
Take me higher.
Can you feel it?

[instrumental break - atmospheric jazz flute improvisation takes the lead over a deep, rolling Detroit house bassline and syncopated acoustic congas]

[chorus - soulful vocal loop]
Lost in the groove tonight...
Just feeling the beat.
Lost in the groove tonight.
Can you feel it?

[climax - full upbeat energy, driving 118 bpm kick, peak jazz flute solo dancing over filtered rhodes chords and pumping house groove]
(fade out)"""

# Define structured output for Replicate UI
class Output(BaseModel):
  audio: Path
  score_abc: str


class Predictor(BasePredictor):

  def setup(self):
    """Loads the official YuE2-3B pipeline onto the GPU once at boot."""
    print("Loading YuE2-3B pipeline from Hugging Face...")
    self.pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda")
    print("YuE2 ready for inference.")

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
      cot: str = Input(
          description=(
              "Symbolic planning: 'full' (plans chords + melody), 'melody'"
              " (plans melody only), 'off' (direct synthesis)"
          ),
          choices=["full", "melody", "off"],
          default="full",
      ),
      cfg_scale: float = Input(
          description="Text prompt guidance scale (1.0 to 1.5 recommended)",
          ge=1.0,
          le=3.0,
          default=1.0,
      ),
      audio_format: str = Input(
          description="Audio output format",
          choices=["mp3", "wav", "flac"],
          default="mp3",
      ),
      custom_abc: str = Input(
          description=(
              "(Optional) Pre-written ABC score. Bypasses the symbolic planner"
              " if provided."
          ),
          default=None,
      ),
      seed: int = Input(
          description="Random seed for reproducibility (-1 for random)",
          default=-1,
      ),
  ) -> Output:
    """Run music generation and return both audio and raw ABC notation."""
    # 1. Random seed resolution
    if seed < 0:
      seed = random.randint(0, 2**32 - 1)
    print(f"Executing generation with seed: {seed}")

    formatted_lyrics = lyrics.replace("\\n", "\n").strip()

    request_kwargs = {
        "style": style,
        "lyrics": formatted_lyrics,
        "cot": cot,
        "cfg_scale": cfg_scale,
        "seed": seed,
    }

    if custom_abc and custom_abc.strip():
      request_kwargs["abc"] = custom_abc.strip()

    # 2. Execute YuE2 pipeline
    song = self.pipe(**request_kwargs)

    # 3. Save artifacts and extract outputs
    output_dir = tempfile.mkdtemp()
    song.save_artifacts(output_dir)

    raw_flac = os.path.join(output_dir, "audio.flac")
    abc_path = os.path.join(output_dir, "score.abc")

    # Read the generated ABC notation if present
    abc_text = ""
    if os.path.exists(abc_path):
      with open(abc_path, "r", encoding="utf-8") as f:
        abc_text = f.read()

    # 4. Transcode audio
    final_audio = raw_flac
    if audio_format == "mp3":
      final_audio = os.path.join(output_dir, "song.mp3")
      subprocess.run(
          ["ffmpeg", "-y", "-i", raw_flac, "-b:a", "320k", final_audio],
          check=True,
      )
    elif audio_format == "wav":
      final_audio = os.path.join(output_dir, "song.wav")
      subprocess.run(
          ["ffmpeg", "-y", "-i", raw_flac, final_audio],
          check=True,
      )

    return Output(audio=Path(final_audio), score_abc=abc_text)