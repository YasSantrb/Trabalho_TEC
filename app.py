import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import re

st.set_page_config(page_title="Detector de Fake News", page_icon="🛡️")
st.title("🛡️ Detector de Fake News - Deep Learning")
st.write("Cole a manchete ou o corpo da notícia abaixo para avaliar a confiabilidade.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_fake_news.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pickle')

def limpar_texto(texto):
    texto = texto.lower() 
    texto = re.sub(r'[^\w\s]', '', texto) 
    return texto

@st.cache_resource
def carregar_componentes():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    modelo = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return modelo, tokenizer

model, tokenizer = carregar_componentes()

if model is None or tokenizer is None:
    st.error("Arquivos do modelo ou tokenizer não encontrados na pasta 'models/'!")
else:
    texto_usuario = st.text_area("Texto da Notícia:", placeholder="Cole aqui o conteúdo da notícia...", height=200)

    if st.button("Verificar Notícia"):
        if texto_usuario.strip() == "":
            st.error("Por favor, digite algum texto antes de verificar.")
        else:
            with st.spinner('Analisando padrões textuais com Deep Learning...'):
                texto_limpo = limpar_texto(texto_usuario)
                
                sequencia = tokenizer.texts_to_sequences([texto_limpo])
                padded = pad_sequences(sequencia, maxlen=300, padding='post', truncating='post')
                
                predicao = model.predict(padded)[0][0]
                
                st.subheader("Resultado da Análise:")
                st.info(f"Valor exato gerado pela IA (Score de 0 a 1): {predicao:.4f}")
                
                if predicao >= 0.5:
                    st.error(f"Alerta: Esta notícia tem {predicao*100:.2f}% de chance de ser FAKE NEWS!")
                else:
                    st.success(f"Confiável: Esta notícia tem {(1-predicao)*100:.2f}% de chance de ser uma NOTÍCIA VERDADEIRA!")