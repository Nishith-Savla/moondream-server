from contextlib import asynccontextmanager
from io import BytesIO

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    CodeGenTokenizerFast as Tokenizer, PreTrainedTokenizerBase,
)

from moondream import Moondream, detect_device

model: Moondream | None = None
tokenizer: PreTrainedTokenizerBase | None = None


def load_model():
    device, dtype = detect_device()

    model_id = "vikhyatk/moondream1"
    tokenizer = Tokenizer.from_pretrained(model_id)
    model = Moondream.from_pretrained(model_id).to(device=device, dtype=dtype)
    model.eval()

    return model, tokenizer


@asynccontextmanager
async def lifespan(_: FastAPI):
    # On startup
    global model, tokenizer
    model, tokenizer = load_model()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def evaluate_image(moondream, tokenizer, image, prompt):
    image_embeds = moondream.encode_image(image)

    return moondream.answer_question(image_embeds, prompt, tokenizer)


@app.post("/evaluate/")
async def evaluate(image: UploadFile = File(...), prompt: str = Form(...)):
    img = Image.open(BytesIO(await image.read()))
    result = evaluate_image(model, tokenizer, img, prompt)
    return {"result": result}


if __name__ == '__main__':
    uvicorn.run("sample:app", host='0.0.0.0', port=8001)
