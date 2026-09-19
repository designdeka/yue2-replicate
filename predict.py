import os
import random
import subprocess
import tempfile
from cog import BasePredictor, Input, Path
from yue2 import YuE2Pipeline

DEFAULT_STYLE = (
    "1980s japanese city pop, upbeat funk groove, funky slap bass, bright brass horns, "
    "sparkling DX7 electric piano, punchy gated reverb drums, crisp rhythm guitar, "
    "120 bpm, nostalgic summer twilight, crystal clear expressive female vocals, breezy disco pop"
)

DEFAULT_LYRICS = """[intro]
(funky slap bass groove, bright brass stabs, sparkling synths)
One, two, cruise into the night!

[verse]
真夜中のベイショア・ルート (Mayonaka no beishoa ruuto)
Neon lights dancing on the passenger window
カーステレオからこぼれるメロディー (Kaa sutereo kara koboreru merodii)
Come and feel the cool night breeze
Speeding down through the Tokyo night
Leave yesterday’s rain behind

[chorus]
Stay with me tonight, 二人だけの (futari dake no)
City lights in the evening glow!
恋は sparkling, running free
もう一度だけ (Mou ichido dake) take a chance with me
Into the summer midnight dream!

[verse]
信号が変わる瞬間に (Shingou ga kawaru shunkan ni)
You turned around with that wistful smile
星屑みたいな街並みを (Hoshikuzu mitai na machinami wo)
We’re chasing shadows for another mile
Don't let the rhythm stop, just let it fly
Under the violet glowing sky

[bridge]
(horns build up with funky guitar chops)
Time is slipping through our fingertips
Can you taste the music on my lips?
Oh, baby don't say goodbye!

[chorus]
Stay with me tonight, 二人だけの (futari dake no)
Glittering city lights in the twilight glow!
恋は sparkling, running free
もう一度だけ (Mou ichido dake) take a chance with me
Into the summer midnight dream!

[outro]
(joyful saxophone solo over driving slap bass)
Night cruise... just you and me.
Forever in the city lights.
(fade out)"""


class Predictor(BasePredictor):
    def setup(self):
        """Loads the official YuE2-3B pipeline onto the GPU once at boot."""
        print("Loading YuE2-3B pipeline from Hugging Face...")
        self.pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda")
        print("YuE2 ready for inference.")

    def run(
        self,
        style: str = Input(
            description="Genre, instruments, mood, tempo, vocal character",
            default=DEFAULT_STYLE,
        ),
        lyrics: str = Input(
            description="Song lyrics with section tags ([verse], [chorus], etc.). Use Shift+Enter for newlines.",
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
            description="(Optional) Supply your own ABC score. Overrides automatic planning if provided.",
            default=None,
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1,
        ),
    ) -> Path:
        """Run music generation using official YuE2 API."""
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        print(f"Executing generation with seed: {seed}")

        formatted_lyrics = lyrics.replace("\\n", "\n").strip()

        # Build official request arguments
        request_kwargs = {
            "style": style,
            "lyrics": formatted_lyrics,
            "cot": cot,
            "cfg_scale": cfg_scale,
            "seed": seed,
        }

        # If user supplied custom ABC score, pass it to bypass the planner
        if custom_abc and custom_abc.strip():
            request_kwargs["abc"] = custom_abc.strip()

        # Execute official pipeline
        song = self.pipe(**request_kwargs)

        output_dir = tempfile.mkdtemp()
        song.save_artifacts(output_dir)
        raw_flac = os.path.join(output_dir, "audio.flac")

        # Transcode output format
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