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
        audio_format: str = Input(
            description="Audio output format (MP3 is compressed and fast to download; WAV/FLAC are lossless)",
            choices=["mp3", "wav", "flac"],
            default="mp3",
        ),
        cot: str = Input(
            description="Symbolic planning: 'full' (chords+melody), 'melody' (faster), 'off' (fastest, skips score)",
            choices=["full", "melody", "off"],
            default="full",
        ),
        diffusion_steps: int = Input(
            description="Acoustic diffusion steps (15-20 for fast drafting, 25-30 for sweet spot, 50 for max fidelity)",
            ge=10,
            le=100,
            default=25,
        ),
        audio_duration: float = Input(
            description="Target song duration in seconds (shorter = faster generation and lower cost)",
            ge=15.0,
            le=360.0,
            default=60.0,
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1,
        ),
    ) -> Path:
        """Run a single music generation request."""
        # 1. Handle random seed: converts -1 into a valid positive 32-bit integer
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        print(f"Executing generation with seed: {seed}")

        # 2. Sanitize lyrics (handles literal '\n' if typed or pasted)
        formatted_lyrics = lyrics.replace("\\n", "\n").strip()

        # 3. Generate music
        song = self.pipe(
            style=style,
            lyrics=formatted_lyrics,
            cot=cot,
            steps=diffusion_steps,
            duration=audio_duration,
            seed=seed,
        )

        output_dir = tempfile.mkdtemp()
        song.save_artifacts(output_dir)
        raw_flac = os.path.join(output_dir, "audio.flac")

        # 4. Transcode to selected format
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