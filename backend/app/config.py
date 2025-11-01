from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List

class Settings(BaseSettings):
    model_camembert_path: str
    model_gpt2_path: str
    default_model: str = "camembert"
    cors_origins: List[str] | str = []

    @field_validator("default_model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        v = v.lower()
        if v not in {"camembert", "gpt2"}:
            raise ValueError("default_model must be 'camembert' or 'gpt2'")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _normalize_cors(cls, v):
        """
        Autorise :
        - chaîne vide => []
        - "*" => ["*"]
        - "http://a, https://b" => ["http://a","https://b"]
        - liste => inchangée
        """
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v == "*":
                return ["*"]
            return [s.strip() for s in v.split(",")]
        return v

    class Config:
        env_prefix = ""
        env_file = ".env"

settings = Settings()
