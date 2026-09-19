import os
import random
import subprocess
import tempfile
from cog import BasePredictor, Input, Path
from yue2 import YuE2Pipeline

DEFAULT_STYLE = (
    "french acid jazz, nu jazz, st germain style, saint germain des pres cafe, "
    "deep jazz house groove, 118 bpm, deep walking acoustic upright bass, "
    "warm fender rhodes electric piano chords, smoky muted jazz trumpet solo, "
    "breathy tenor saxophone, atmospheric jazz flute, syncopated congas and brushed drums, "
    "late night parisian lounge, pure instrumental, sophisticated, vinyl warmth"
)

DEFAULT_LYRICS = """[intro - warm vinyl static, soft finger snaps, brushed ride cymbal, deep acoustic upright bass walking groove]

[instrumental - gentle fender rhodes electric piano comping minor ninth chords with lush vintage tremolo]

[theme - muted jazz trumpet enters playing a smoky, melodic motif over a steady 118 bpm deep house kick]

[variation - upright bass intensifies with walking swing syncopation, acoustic congas and light shakers enter]

[solo - breathy tenor saxophone takes over the lead with bluesy, expressive jazz runs]

[breakdown - kick drum drops out, warm rhodes chords sustain in reverb, airy flute improvisations drift over vinyl crackle]

[buildup - four-on-the-floor brushed kick re-enters, upright bass locks back into the deep pocket groove]

[climax - muted trumpet and tenor sax harmonize over the central riff, syncopated jazz percussion peaks]

[interlude - extended rhodes electric piano solo with subtle stereo panning and complex chord voicings]

[reprise - muted trumpet plays the opening melodic theme softly over the hypnotic walking bassline]

[outro - trumpet and sax fade out, leaving only the upright double bass, Rhodes chords, and gentle brush percussion slowly dissolving into vinyl static]"""


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
            description="Song lyrics or bracketed musical structure tags ([intro], [solo], etc.)",
            default=DEFAULT_LYRICS,
        ),
        cot: str = Input(
            description="Symbolic planning: 'full' (plans chords + melody), 'melody' (plans melody only), 'off' (direct synthesis)",
            choices=["full", "melody", "off"],
            default="full",
        ),
        cfg_scale: float = Input(
            description="Text prompt guidance scale (1.0 to 1.5 recommended)",
            ge=1.0,
            le=3.0,
            default=1.2,
        ),
        audio_format: str = Input(
            description="Audio output format",
            choices=["mp3", "wav", "flac"],
            default="mp3",
        ),
        custom_abc: str = Input(
            description="(Optional) Pre-written ABC score. Bypasses the symbolic planner if provided.",
            default=None,
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1,
        ),
    ) -> Path:
        """Run music generation using official YuE2 API."""
        # 1. Handle random seed: converts -1 into a valid positive 32-bit integer
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        print(f"Executing generation with seed: {seed}")

        # 2. Convert literal '\n' if typed into real newlines
        formatted_lyrics = lyrics.replace("\\n", "\n").strip()

        # 3. Assemble parameters strictly accepted by SongRequest
        request_kwargs = {
            "style": style,
            "lyrics": formatted_lyrics,
            "cot": cot,
            "cfg_scale": cfg_scale,
            "seed": seed,
        }

        if custom_abc and custom_abc.strip():
            request_kwargs["abc"] = custom_abc.strip()

        # 4. Execute official pipeline
        song = self.pipe(**request_kwargs)

        # 5. Save generated artifacts
        output_dir = tempfile.mkdtemp()
        song.save_artifacts(output_dir)
        raw_flac = os.path.join(output_dir, "audio.flac")

        # 6. Transcode to requested format
        if audio_format == "mp3":
            final_output = os.path.join(output_dir, "song.mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_flac, "-b:a", "320k", final_output],
                check=True,
            )
            return Path(final_output)

        elif audio_format == "wav":
            final_output = os.path.join(output_dir, "song.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_flac, final_output],
                check=True,
            )
            return Path(final_output)

        return Path(raw_flac)