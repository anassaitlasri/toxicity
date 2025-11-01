import numpy as np
import torch
from ..model_registry import registry
from lime.lime_text import LimeTextExplainer
import shap
from captum.attr import IntegratedGradients

def explain_lime(text: str, model_name: str | None, num_features=10, num_samples=1000, max_length=128):
    m = registry.get(model_name)
    explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"])

    def _predict(batch: list[str]):
        out = []
        for t in batch:
            enc = m.tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(m.device)
            with torch.no_grad():
                logits = m.model(**enc).logits
                prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            out.append(prob)
        return np.array(out)

    exp = explainer.explain_instance(text, _predict, num_features=num_features, num_samples=num_samples)
    html = exp.as_html()
    return {"html": html}

def explain_shap(text: str, model_name: str | None):
    m = registry.get(model_name)
    explainer = shap.Explainer(m.pipeline)
    shap_values = explainer([text])[0]
    html = shap.plots.text(shap_values, display=False)
    return {"html": html}

def explain_ig(text: str, model_name: str | None, max_length=128):
    m = registry.get(model_name)
    enc = m.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
    input_ids = enc["input_ids"].to(m.device)
    att_mask  = enc["attention_mask"].to(m.device)

    if m.name == "camembert":
        embeddings = m.model.roberta.embeddings(input_ids)
    elif m.name == "gpt2":
        embeddings = m.model.transformer.wte(input_ids)
    else:
        raise RuntimeError("Unsupported model for IG")

    def forward_func(emb, mask):
        out = m.model(inputs_embeds=emb, attention_mask=mask)
        return torch.softmax(out.logits, dim=1)[:, 1]

    ig = IntegratedGradients(forward_func)
    baseline = torch.zeros_like(embeddings)
    attributions, _ = ig.attribute(inputs=embeddings, baselines=baseline, additional_forward_args=(att_mask,), return_convergence_delta=True)

    tokens = m.tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attributions.sum(dim=-1).squeeze(0)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    pairs = [{"token": tok, "score": float(s)} for tok, s in zip(tokens, scores) if tok not in {"<pad>", "<s>", "</s>", "<|endoftext|>"}]
    return {"attributions": pairs}
