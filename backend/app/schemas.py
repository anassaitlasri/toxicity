from pydantic import BaseModel, Field
from typing import Literal, List

ModelName = Literal["camembert", "gpt2"]

class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    model: ModelName | None = None
    max_length: int = 128

class ClassProb(BaseModel):
    label: Literal["non-toxic", "toxic"]
    score: float

class PredictResponse(BaseModel):
    model: ModelName
    probs: List[ClassProb]
    toxic_score: float

class ExplainRequest(PredictRequest):
    method: Literal["lime", "shap", "ig"]
    num_features: int | None = 10
    num_samples: int | None = 1000

class TokenAttribution(BaseModel):
    token: str
    score: float

class ExplainResponse(BaseModel):
    model: ModelName
    method: str
    toxic_score: float
    attributions: List[TokenAttribution] | None = None
    html: str | None = None
