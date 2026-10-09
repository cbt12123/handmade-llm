"""Single-process local HTTP service; one model instance and fail-fast admission."""
from contextlib import asynccontextmanager
import logging
import threading
import uuid
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from _common import load_model, encode_messages, generate, MODEL_ID

logger = logging.getLogger("uvicorn.error")
gate = threading.Lock()
MAX_CONTEXT = 512
MAX_OUTPUT = 128


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class GenerateRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_new_tokens: int = Field(default=64, ge=1, le=MAX_OUTPUT)

    @model_validator(mode="after")
    def check_order(self):
        roles = [m.role for m in self.messages]
        start = 1 if roles[0] == "system" else 0
        expected = ["user" if i % 2 == 0 else "assistant" for i in range(len(roles)-start)]
        if roles[start:] != expected or roles[-1] != "user":
            raise ValueError("Optional first system message, then user/assistant alternating, ending with user")
        return self


@asynccontextmanager
async def lifespan(app):
    model, tokenizer, load_seconds = load_model()
    app.state.model, app.state.tokenizer = model, tokenizer
    app.state.load_seconds = load_seconds
    yield
    del app.state.model


app = FastAPI(title="Local LLM teaching service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready":True, "model":MODEL_ID, "device":str(app.state.model.device),
            "busy":gate.locked(), "context_limit":MAX_CONTEXT, "output_limit":MAX_OUTPUT,
            "load_seconds":app.state.load_seconds}


@app.post("/generate")
def generate_text(request: GenerateRequest):
    # Admission includes tokenization, so concurrent requests do not touch one tokenizer instance.
    if not gate.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Model busy; retry after the current request finishes")
    request_id = uuid.uuid4().hex
    try:
        messages = [m.model_dump() for m in request.messages]
        inputs = encode_messages(app.state.tokenizer, messages)
        input_tokens = int(inputs.input_ids.shape[1])
        if input_tokens + request.max_new_tokens > MAX_CONTEXT:
            raise HTTPException(status_code=413, detail="Input tokens plus max_new_tokens exceed the context limit")
        result = generate(app.state.model, app.state.tokenizer, messages, request.max_new_tokens)
        generated = result.pop("token_ids")
        eos = app.state.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        reason = "eos" if generated and generated[-1] in eos_ids else "length"
        logger.info("request_id=%s input_tokens=%d output_tokens=%d seconds=%.3f",
                    request_id, result["input_tokens"], result["output_tokens"], result["generation_seconds"])
        return {"request_id":request_id, "model":MODEL_ID, "finish_reason":reason, **result}
    except HTTPException:
        raise
    except Exception:
        logger.exception("generation failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail={"message":"Generation failed; inspect local server logs",
                                                   "request_id":request_id})
    finally:
        gate.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
