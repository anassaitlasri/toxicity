import numpy as np
import torch
from ..model_registry import registry

def _softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

def predict(text: str, model_name: str | None, max_length: int = 128):
    m = registry.get(model_name)
    inputs = m.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(m.device)
    with torch.no_grad():
        logits = m.model(**inputs).logits
    probs = _softmax(logits)
    return {
        "model": m.name,
        "probs": [{"label": "non-toxic", "score": float(probs[0])},
                  {"label": "toxic",     "score": float(probs[1])}],
        "toxic_score": float(probs[1]),
    }
