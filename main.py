from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

@app.post("/analisar")
async def analisar(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=80)
    caption = processor.decode(out[0], skip_special_tokens=True)
    prompt_final = f"Imagem de IA: {caption}. Descrição gerada automaticamente a partir da análise visual."
    return {"prompt": prompt_final}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
