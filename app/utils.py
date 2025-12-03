import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Configuração Global
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def get_model():
    """
    Recria a arquitetura da ResNet50 e carrega os pesos treinados.
    """
    model = models.resnet50(weights=None)
    
    num_ftrs = model.fc.in_features # 2048
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 6)
    )
    
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
        print(f"Modelo carregado com sucesso no dispositivo: {DEVICE}")
    except FileNotFoundError:
        print("Erro: Arquivo 'models/best_model.pth' não encontrado.")
        raise

    model.to(DEVICE)
    model.eval() # Modo de inferência (desliga Dropout/BatchNorm)
    return model

def transform_image(image_bytes):
    """
    Recebe os bytes crus da imagem (do upload), converte para PIL
    e aplica as transformações necessárias para a ResNet.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Cria um fundo branco do mesmo tamanho
        alpha = image.convert('RGBA').split()[-1]
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg
    else:
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(DEVICE)