from fastapi import FastAPI, File, UploadFile, HTTPException
from app.utils import get_model, transform_image, CLASS_NAMES
import torch
import uvicorn


app = FastAPI(title="EcoSort AI API", description="Classificador de Resíduos com Deep Learning")

# Variável global para guardar o modelo em memória
model = None

@app.on_event("startup")
async def load_ai_model():
    """Carrega o modelo apenas uma vez quando o servidor inicia."""
    global model
    model = get_model()

@app.get("/")
def read_root():
    return {"status": "online", "message": "EcoSort AI is running! ♻️"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint que recebe uma imagem e retorna a classe do lixo.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem.")
    
    try:
        # Le os bytes da imagem
        image_bytes = await file.read()
        
        # Pré-processamento
        tensor = transform_image(image_bytes)
        
        # Inferência (Passar pelo modelo)
        with torch.no_grad():
            outputs = model(tensor)
            # Softmax para ter probabilidades (0 a 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
        
        class_name = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return {
            "filename": file.filename,
            "class": class_name,
            "confidence": f"{confidence_score:.2%}",
            "confidence_score": confidence_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)