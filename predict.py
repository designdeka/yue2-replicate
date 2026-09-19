import os
import tempfile
from cog import BasePredictor, Input, Path
from yue2 import YuE2Pipeline

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
            default="1980s japanese city pop, upbeat funk groove, slap bass, bright brass, 120 bpm, female vocals"
        ),
        lyrics: str = Input(
            description="Song lyrics with section tags ([verse], [chorus], etc.)",
            default="[verse]\nCity lights shining in the night...\n[chorus]\nStay with me tonight!"
        ),
        cot: str = Input(
            description="Planning mode: full (melody + chords), melody (melody only), off (no score)",
            choices=["full", "melody", "off"],
            default="full"
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1
        ),
    ) -> Path:
        """Run a single music generation request."""
        gen_seed = None if seed == -1 else seed
        
        song = self.pipe(
            style=style,
            lyrics=lyrics,
            cot=cot,
            seed=gen_seed,
        )
        
        output_dir = tempfile.mkdtemp()
        song.save_artifacts(output_dir)
        
        output_flac = os.path.join(output_dir, "audio.flac")
        return Path(output_flac)