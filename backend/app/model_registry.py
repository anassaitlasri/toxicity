from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .config import settings
import torch

class ModelBundle:
    def __init__(self, name: str, path: str, device: torch.device):
        self.name = name
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self.model.to(device).eval()
        self.device = device
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )

class Registry:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, ModelBundle] = {}

    def load_all(self) -> None:
        self.models["camembert"] = ModelBundle("camembert", settings.model_camembert_path, self.device)
        self.models["gpt2"]      = ModelBundle("gpt2", settings.model_gpt2_path, self.device)

    def get(self, name: str | None) -> ModelBundle:
        key = (name or settings.default_model).lower()
        if key not in self.models:
            raise KeyError(f"Unknown model: {key}")
        return self.models[key]

registry = Registry()
