from fastapi import APIRouter, HTTPException
from ..schemas import PredictRequest, PredictResponse, ExplainRequest, ExplainResponse
from ..services import inference, explanations
import anyio

router = APIRouter(prefix="/v1", tags=["toxicity"])

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        result = await anyio.to_thread.run_sync(inference.predict, req.text, req.model, req.max_length)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    try:
        pred = await anyio.to_thread.run_sync(inference.predict, req.text, req.model, req.max_length)
        if req.method == "lime":
            out = await anyio.to_thread.run_sync(explanations.explain_lime, req.text, req.model, req.num_features, req.num_samples, req.max_length)
            return ExplainResponse(model=pred["model"], method="lime", toxic_score=pred["toxic_score"], html=out["html"])
        if req.method == "shap":
            out = await anyio.to_thread.run_sync(explanations.explain_shap, req.text, req.model)
            return ExplainResponse(model=pred["model"], method="shap", toxic_score=pred["toxic_score"], html=out["html"])
        out = await anyio.to_thread.run_sync(explanations.explain_ig, req.text, req.model, req.max_length)
        return ExplainResponse(model=pred["model"], method="ig", toxic_score=pred["toxic_score"], attributions=out["attributions"])
    except Exception as e:
        raise HTTPException(400, str(e))
