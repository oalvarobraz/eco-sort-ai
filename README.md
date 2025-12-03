# ♻️ EcoSort AI: Classificador de Resíduos com Transfer Learning

> **Status:** Concluído ✅  
> **Modelo:** ResNet50 (Transfer Learning)  
> **Acurácia Final:** 90.79% (TrashNet)  
> **Arquitetura:** MLOps Pipeline Completo (Model + API + Frontend)

O **EcoSort AI** é uma solução completa de Visão Computacional para automação da reciclagem. O projeto vai além do treinamento do modelo, implementando um **pipeline de MLOps** com API de inferência e interface de usuário, focando em robustez contra viés de contexto.

O sistema aborda desafios reais de engenharia, como **datasets desbalanceados**, **viés de contexto** (background bias) e **training-serving skew**, demonstrando a diferença entre performance em ambiente controlado e aplicação no mundo real.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![ResNet](https://img.shields.io/badge/Model-ResNet50-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/TrashNet-SOTA-success?style=for-the-badge)

---

## 🏗️ Arquitetura do Sistema

O projeto foi estruturado simulando um **ambiente de produção real** com três componentes integrados:

### 🔹 Stack Completa

1. **Núcleo de IA (PyTorch):** Treinamento com Transfer Learning e correção de desbalanceamento
2. **Backend (FastAPI):** API RESTful que serve o modelo, tratando imagens (bytes) e normalização
3. **Frontend (Streamlit):** Interface web amigável para upload e classificação em tempo real

---

## 🗂️ Estrutura do Projeto

```text
waste-classifier-pytorch/
│
├── app/
│   ├── main.py          # API FastAPI
│   ├── frontend.py      # Interface Streamlit
│   └── utils.py         # Lógica de Inferência e Pré-processamento
│
├── src/
│   └── notebook.ipynb   # Treinamento e Análise
│
├── models/              # Modelos salvos
│
├── assets/              # Imagens de demonstração e matriz de confusão
│
├── TrashNet/            # Dataset (baixar separadamente)
│
├── requirements.txt     # Dependências do projeto
│
└── README.md            # Este arquivo
```

---

## 🛠️ Tecnologias Utilizadas

### 🔹 Machine Learning

- **PyTorch:** Construção do modelo e treinamento
- **Torchvision:** Transfer Learning e transformações
- **Scikit-learn:** Métricas de avaliação e split estratificado

### 🔹 Backend & API

- **FastAPI:** API RESTful de alta performance
- **Uvicorn:** Servidor ASGI para FastAPI
- **Pillow:** Processamento de imagens

### 🔹 Frontend

- **Streamlit:** Interface web interativa
- **Requests:** Comunicação com a API

### 🔹 Visualização & Análise

- **Matplotlib:** Visualizações e matriz de confusão
- **Pandas/NumPy:** Manipulação de dados

---

## 📊 Sobre o Dataset (TrashNet)

Utilizei o dataset padrão da indústria, **TrashNet**, contendo 2.527 imagens divididas em 6 categorias.

### 🔹 Classes

- 📦 **Cardboard** (Papelão)
- 🍷 **Glass** (Vidro)
- 🔩 **Metal**
- 📄 **Paper** (Papel)
- 🧴 **Plastic** (Plástico)
- 🗑️ **Trash** (Lixo Geral)

### 🔹 Desafio: Dataset Desbalanceado

O dataset original apresenta forte desbalanceamento (muito papel, pouco lixo geral), o que pode causar viés no modelo.

#### 📌 Soluções Implementadas:

**1. Split Estratificado**
- Mantém a proporção de classes em Treino/Validação/Teste
- Garante que todas as classes estejam representadas adequadamente

**2. Balanceamento via Pesos**
- `WeightedRandomSampler` durante o treinamento
- Pesos na Loss Function calculados inversamente à frequência das classes
- Penaliza mais o modelo quando erra classes minoritárias (como `Trash`)

---

## 🧠 Arquitetura do Modelo

Utilizei **Transfer Learning** para contornar a escassez de dados e acelerar a convergência.

### 🔹 Backbone

- **ResNet50** pré-treinada na ImageNet
- Feature Extractor congelado (pesos mantidos fixos)
- Aproveita representações visuais aprendidas de 1.4 milhões de imagens

### 🔹 Classificador Customizado

```python
classifier = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),        # Crucial para regularização
    nn.Linear(1024, 6)      # 6 classes de resíduos
)
```

**Dropout (0.5)** foi essencial para evitar overfitting dado o tamanho reduzido do dataset.

---

## 📈 Resultados em Ambiente Controlado

O modelo atingiu resultados competitivos com o Estado da Arte para este dataset.

| Métrica | Valor |
|---------|-------|
| **Acurácia (Teste)** | **90.53%** |
| **Épocas Treinadas** | 11 (Early Stopping) |
| **Tempo de Treino** | ~10 minutos (GPU T4) |

### 🔹 Análise da Matriz de Confusão
Abaixo, a Matriz de Confusão mostrando os acertos por classe:

<div align="center">
  <img src="./assets/confusion_matrix.png" alt="Matriz de Confusão" width="600">
  <p><em>Figura 1: Matriz de Confusão do modelo no conjunto de teste</em></p>
</div>

**Pontos Fortes:**
- ✅ **Trash (Lixo Geral):** 19/20 acertos - estratégia de pesos funcionou perfeitamente
- ✅ Excelente distinção entre **Papel** e **Papelão**

**Desafios Identificados:**
- ⚠️ Leve confusão entre **Vidro** e **Plástico** devido à transparência e reflexos similares (esperado e reduzido)
- ⚠️ Algumas confusões em materiais com textura ambígua

---

## 🧪 Case Study: O Desafio do Mundo Real

A maior conquista deste projeto foi **corrigir o Training-Serving Skew**. Após validar o modelo com 90% de precisão no dataset (fundo branco uniforme), realizamos testes com **fotos reais de smartphone** para avaliar a robustez em condições não controladas.

### 🚨 Descoberta: Viés de Contexto Significativo

**Problema detectado:** O modelo aprendeu a associar o **fundo** da imagem à classe, não apenas o objeto.

**Solução implementada:** Aplicação de **Normalização da ImageNet** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) tanto no treino quanto na inferência, forçando o modelo a focar em características do objeto.
---

### 📱 Caso 1: O "Papelão" de Madeira

**Experimento Inicial (Antes da Correção):**
- Fotografamos uma **folha de papel branca** sobre um **piso de madeira**

**Resultado Anterior:**
- 🏷️ **Classe Real:** Paper
- 🧠 **Predição:** `CARDBOARD` (99.9% de confiança) ❌

**Diagnóstico:**
- O modelo associou a cor marrom e textura do chão à classe Papelão
- Ignorou completamente a cor branca e textura lisa do papel
- **Shortcut learning:** Aprendeu atalho visual (fundo) em vez da característica real (objeto)

**Após Correção:**
- Sistema agora aplica normalização adequada na inferência
- Viés de fundo drasticamente reduzido ✅

---

### 🌿 Caso 2: A Garrafa de Plástico na Grama

**Experimento (Teste Extremo):**
- Fotografamos uma **garrafa de plástico transparente** jogada na **grama** (fundo nunca visto no treino)

**Resultado:**
- 🏷️ **Classe Real:** Plastic
- 🧠 **Predição:** `PLASTIC` (62.18% de confiança) ✅

**Análise:**
- A confiança foi menor (o que é **honesto e esperado**)
- A decisão foi **correta**
- O modelo aprendeu a forma do objeto, **ignorando o fundo verde**
- Demonstra robustez contra ambientes não controlados

---

### 🍾 Caso 3: O Vidro com Fundo Complexo

**Experimento:**
- Fotografamos uma **garrafa de vidro** em ambiente complexo com fundo variado

**Resultado Inicial (Antes da Correção):**
- 🏷️ **Classe Real:** Glass
- 🧠 **Predição:** `PLASTIC` (83% de confiança) ❌
- **Problema:** Modelo confundia reflexos com características de plástico

**Resultado Atual (Após Correção):**
- 🧠 **Predição (Fundo Complexo):** `GLASS` (81.93% de confiança) ✅
- 🧠 **Predição (Fundo Branco):** `GLASS` (98.46% de confiança) ✅

**Conclusão:**
- O modelo mantém coerência da classificação em ambientes não controlados
- Normalização correta foi **crítica** para generalização

---

## 🔬 Análise Técnica do Viés e Solução

### Por que o viés acontecia?

1. **Dataset Homogêneo:** TrashNet possui todas as imagens com fundo branco/neutro
2. **Feature Learning Incorreto:** A rede aprendia que fundo marrom/texturizado = Papelão
3. **Ausência de Variabilidade:** Não há exemplos de papel em fundos escuros ou papelão em fundos claros
4. **Training-Serving Skew:** Diferença entre pré-processamento no treino vs. inferência

### Como corrigi?

**Solução Principal: Normalização da ImageNet**
```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # Valores da ImageNet
    std=[0.229, 0.224, 0.225]
)
```

**Aplicação consistente em:**
- ✅ Pipeline de treinamento
- ✅ Pipeline de validação
- ✅ API de inferência (FastAPI)
- ✅ Frontend (Streamlit)

**Resultado:** Redução drástica do viés de fundo, permitindo generalização para ambientes reais.

## 🚀 Roadmap: Próximos Passos para Produção

Para tornar o EcoSort AI ainda mais robusto para aplicação em larga escala:

### 🔹 1. Segmentação Prévia

Implementar um modelo de segmentação (ex: **U-Net** ou **Mask R-CNN**) para:
- Isolar o objeto do fundo
- Remover background antes da classificação
- Pipeline de dois estágios: Segmentação → Classificação

### 🔹 2. Data Augmentation Avançado

- **Mosaic Augmentation:** Inserir fundos aleatórios durante o treino
- **CutOut:** Mascarar partes da imagem para forçar o modelo a não depender de contexto
- **MixUp:** Misturar imagens de diferentes classes

### 🔹 3. Dataset Expandido

- Coletar imagens com fundos diversos (madeira, concreto, grama, etc.)
- Incluir variações de iluminação (dia, noite, sombra)
- Adicionar objetos em diferentes ângulos e distâncias

### 🔹 4. Domain Adaptation

- Treinar em imagens sintéticas com fundos variados
- Aplicar técnicas de **Domain Randomization**
- Fine-tuning em dados reais coletados de usuários

### 🔹 5. Deploy em Produção

- Containerização com **Docker**
- Deploy em nuvem (AWS/GCP/Azure)
- Monitoramento de drift do modelo
- Sistema de feedback para retreinamento contínuo

---

## ⚙️ Como Executar

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/oalvarobraz/eco-sort-ai.git
cd eco-sort-ai
```

### 2️⃣ Baixe o dataset

Acesse o [TrashNet Dataset](https://www.kaggle.com/datasets/feyzazkefe/trashnet/code) e extraia `TrashNet/`

### 3️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Execute o notebook (opcional)

```bash
jupyter notebook src/notebook.ipynb
```

### 5️⃣ Inicie o Backend (API)

Em um terminal, inicie o servidor FastAPI:

```bash
uvicorn app.main:app --reload
```

O servidor rodará em `http://127.0.0.1:8000`

### 6️⃣ Inicie o Frontend

Em outro terminal, inicie o Streamlit:

```bash
streamlit run app/frontend.py
```

O navegador abrirá automaticamente. Basta arrastar uma imagem para classificar! 🎉

---

## 📦 Dependências Principais

```text
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Backend
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# Frontend
streamlit>=1.28.0
requests>=2.31.0

# Processamento
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0

# Visualização
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## ✅ Principais Aprendizados

- **Transfer Learning:** ResNet50 acelerou drasticamente o treinamento e melhorou a generalização
- **Balanceamento de Classes:** Pesos na loss function foram cruciais para lidar com desbalanceamento
- **Training-Serving Skew:** Normalização consistente entre treino e inferência é **crítica**
- **Viés de Dataset:** Identificação prática de shortcut learning e suas implicações em produção
- **Gap Lab → Real:** Métricas em ambiente controlado não garantem performance em aplicação real
- **MLOps Pipeline:** Implementação de API + Frontend simula ambiente de produção real
- **Confiança do Modelo:** Importante monitorar não apenas a classe predita, mas também a confiança

---

## 🎯 Conclusão

O **EcoSort AI** demonstra tanto o **potencial** quanto as **limitações** de Deep Learning aplicado a problemas reais, e como superá-las através de engenharia cuidadosa.

**Conquistas:**
- ✅ 90.79% de acurácia em ambiente controlado
- ✅ Robustez em ambientes não controlados após correção de viés
- ✅ Pipeline MLOps completo (Modelo + API + Frontend)
- ✅ Identificação e correção de training-serving skew

**Lições Críticas:**
1. **Dataset diversificado** que represente o ambiente de produção
2. **Pré-processamento consistente** entre treino e inferência
3. **Validação além das métricas** - testar em cenários não controlados
4. **Monitoramento de confiança** - não apenas a classe predita

Este projeto serve como estudo de caso valioso sobre a diferença entre **validação de laboratório** e **robustez no mundo real**, e como construir sistemas de ML verdadeiramente confiáveis.

---

## 📌 Autor

**Álvaro Braz**

Projeto desenvolvido para fins de **estudo, pesquisa e portfólio profissional em Visão Computacional, Deep Learning e MLOps**.

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 🙏 Agradecimentos

- **TrashNet Dataset:** Gary Thung & Mindy Yang
- **PyTorch Team:** Pela framework excepcional
- **FastAPI & Streamlit:** Por tornarem deploy de ML acessível
- **Comunidade de Deep Learning:** Pelas discussões sobre domain adaptation e robustez de modelos