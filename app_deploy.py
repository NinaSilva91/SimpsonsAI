import os
import json
import io
import torch

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelo = torch.jit.load(
    os.path.join(BASE_DIR, "modelo_simpsons.pt"),
    map_location=device
)
modelo.eval()

with open(os.path.join(BASE_DIR, "clases.json"), encoding="utf-8") as f:
    clases = json.load(f)

transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

app = FastAPI(title="API Simpson Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def inicio():
    return {"mensaje": "API de Los Simpson funcionando"}

@app.get("/classes")
def obtener_clases():
    return {"clases": clases}

@app.post("/predict")
async def predecir(file: UploadFile = File(...)):
    try:
        contenido = await file.read()
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
        tensor = transformacion(imagen).unsqueeze(0).to(device)

        with torch.no_grad():
            salidas = modelo(tensor)
            probabilidades = torch.softmax(salidas, dim=1)
            confianza, indice = probabilidades.max(dim=1)

        return {
            "personaje": clases[indice.item()],
            "confianza": round(confianza.item() * 100, 2)
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


from fastapi.responses import FileResponse

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


@app.get("/web")
def pagina_web():
    return FileResponse(str(BASE_DIR / "index.html"))
