import os
import torch
import torchaudio
from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# The upstream YuE 2 checkpoint
MODEL_ID = "m-a-p/YuE2" 

class Predictor(BasePredictor):
    def setup(self):
        """Runs once when the cloud GPU container starts up."""
        print("Loading YuE2 weights onto GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Native BF16 precision (runs at full speed on A40/A100)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("YuE2 ready for inference.")

    def predict(
        self,
        style_prompt: str = Input(
            description="Genre, mood, tempo, instruments, vocal style",
            default="lofi hip hop, jazzy boom bap, warm Rhodes piano, relaxed 86 bpm swing drums, introspective male vocals"
        ),
        lyrics: str = Input(
            description="Lyrics with structure tags ([verse], [chorus], etc.)",
            default="[verse]\nRaindrops tapping on the glass...\n[chorus]\nTime to let the cadence pass."
        ),
        max_tokens: int = Input(
            description="Maximum generation tokens",
            default=1500
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=-1
        ),
    ) -> Path:
        """Runs per request."""
        if seed != -1:
            torch.manual_seed(seed)
            
        output_file = "/tmp/song_output.mp3"

        # --- Ingest Prompt & Run Generation ---
        # (Execute model generation pipeline and save output_file)

        return Path(output_file)