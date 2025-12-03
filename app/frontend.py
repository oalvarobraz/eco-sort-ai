import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="centered")

# URL da sua API (Backend)
API_URL = "http://127.0.0.1:8000/predict"


st.title("♻️ EcoSort AI")
st.markdown("### Seu Assistente Inteligente de Reciclagem")
st.info("Faça upload de uma foto de lixo (papel, plástico, vidro...) e a IA dirá onde descartar!")


uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem enviada", width='stretch')
    

    if st.button("🔍 Classificar Lixo"):
        with st.spinner("Analisando..."):
            try:
                # Envia para a API (Backend)
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                
                response = requests.post(API_URL, files=files)
                
                # Processa o Resultado
                if response.status_code == 200:
                    result = response.json()
                    label = result['class'].upper()
                    confidence = result['confidence_score'] * 100
                    
                    # Mostra o Resultado
                    st.success("Análise Concluída!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Material Detectado", value=label)
                    with col2:
                        st.metric(label="Confiança", value=f"{confidence:.2f}%")
                    
                    # Dica de Descarte
                    if label in ['PAPER', 'CARDBOARD']:
                        st.info("🟦 **Descarte:** Lixeira AZUL (Papel)")
                    elif label == 'PLASTIC':
                        st.warning("🟥 **Descarte:** Lixeira VERMELHA (Plástico)")
                    elif label == 'GLASS':
                        st.success("🟩 **Descarte:** Lixeira VERDE (Vidro)")
                    elif label == 'METAL':
                        st.warning("🟨 **Descarte:** Lixeira AMARELA (Metal)")
                    else:
                        st.error("⬛ **Descarte:** Lixeira CINZA (Não Reciclável/Orgânico)")
                        
                else:
                    st.error(f"Erro na API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Erro de conexão: {e}. Verifique se o Backend está rodando!")

st.markdown("---")
st.caption("Desenvolvido com PyTorch, FastAPI e Streamlit")