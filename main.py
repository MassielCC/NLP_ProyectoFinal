import pandas as pd
import streamlit as st
from datetime import datetime
from copy import deepcopy
import openai
import csv
import re
import pytz
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar la clave API de OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Configuración inicial de la página
st.set_page_config(page_title="Nova-Infor", page_icon=":computer:")
st.title("👨‍💻 Nova-Infor")

# Mensaje de bienvenida
intro = """¡Bienvenido a Nova-Infor, tu consejero virtual en Ingeniería Informática."""
st.markdown(intro)

# Cargar y procesar los datos de maestros
def load_maestros(file_path):
    """Cargar datos de maestros desde un archivo CSV y convertirlo en un DataFrame."""
    maestros_df = pd.read_csv(file_path)
    maestros_df = maestros_df.fillna('')
    return maestros_df

# Cargar y procesar los datos de estudiantes
def load_estudiantes(file_path):
    """Cargar datos de estudiantes desde un archivo CSV y convertirlo en un DataFrame."""
    estudiantes_df = pd.read_csv(file_path)
    estudiantes_df = estudiantes_df.fillna('')
    return estudiantes_df

# Cargar los datos
maestros_df = load_maestros("Entrevistas_maestros.csv")
estudiantes_df = load_estudiantes("Entrevistas_estudiantes.csv")

# Combinar las preguntas y respuestas en una lista para vectorización
def preparar_datos(df):
    """Preparar datos para la búsqueda basada en similitud de coseno."""
    datos = []
    for index, row in df.iterrows():
        for col in df.columns:
            if col != 'Pregunta' and row[col].strip() != '':
                datos.append({
                    'pregunta': row['Pregunta'],
                    'respuesta': row[col],
                    'origen': col  # Nombre del profesor o estudiante
                })
    return datos

maestros_data = preparar_datos(maestros_df)
estudiantes_data = preparar_datos(estudiantes_df)
todos_los_datos = maestros_data + estudiantes_data

# Crear vectorizador TF-IDF y ajustar con las preguntas y respuestas
vectorizer = TfidfVectorizer()
corpus = [dato['pregunta'] + ' ' + dato['respuesta'] for dato in todos_los_datos]
X = vectorizer.fit_transform(corpus)

def buscar_respuesta(prompt):
    """Buscar la respuesta más relevante en los datos utilizando similitud de coseno."""
    prompt_vector = vectorizer.transform([prompt])
    similitudes = cosine_similarity(prompt_vector, X).flatten()
    max_similitud = similitudes.max()
    if max_similitud > 0.3:  # Umbral de similitud
        index = similitudes.argmax()
        respuesta = todos_los_datos[index]['respuesta']
        origen = todos_los_datos[index]['origen']
        pregunta_original = todos_los_datos[index]['pregunta']
        return respuesta, origen, pregunta_original
    else:
        return None, None, None

def get_system_prompt():
    """Define el prompt del sistema para el chatbot."""
    system_prompt = """
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática.
    Tu objetivo es ayudar a los estudiantes a descubrir su especialidad ideal dentro de la carrera.
    Utiliza las respuestas de los profesores y estudiantes proporcionadas en los datos.
    Si no tienes una respuesta directa en tus datos, proporciona una respuesta general y útil.
    """
    return system_prompt

def generate_response(prompt, temperature=0.5, max_tokens=1000):
    """Generar una respuesta basada en los datos o utilizando OpenAI."""
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Intentar encontrar una respuesta en los datos
    respuesta, origen, pregunta_original = buscar_respuesta(prompt)

    if respuesta:
        # Si se encuentra una respuesta, devolverla
        response = f"*{origen}* responde a tu consulta relacionada con *'{pregunta_original}'*:\n\n{respuesta}"
        st.session_state["messages"].append({"role": "assistant", "content": response})
        return response
    else:
        # Si no se encuentra respuesta, usar OpenAI
        messages = st.session_state["messages"]
        messages.insert(0, {"role": "system", "content": get_system_prompt()})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = completion.choices[0].message.content
        st.session_state["messages"].append({"role": "assistant", "content": response})
        return response

# Función para verificar contenido inapropiado
def check_for_inappropriate_content(prompt):
    """Verifica si el prompt contiene contenido inapropiado utilizando la API de Moderación de OpenAI."""
    try:
        response = openai.Moderation.create(input=prompt)
        logging.info(f"Moderation API response: {response}")
        moderation_result = response["results"][0]
        if moderation_result["flagged"]:
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error al llamar a la API de Moderación: {e}")
        return False

# Estado inicial de la conversación
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": get_system_prompt()},
        {
            "role": "assistant",
            "content": "¡Hola! Soy tu asistente virtual para ayudarte a elegir la especialidad ideal en Ingeniería Informática. ¿En qué puedo ayudarte hoy?",
        },
    ]

# Botón para eliminar conversación
clear_button = st.button("Eliminar conversación", key="clear")
if clear_button:
    st.session_state["messages"] = [
        {"role": "system", "content": get_system_prompt()},
        {
            "role": "assistant",
            "content": "¡Hola de nuevo! Estoy listo para ayudarte en lo que necesites sobre Ingeniería Informática.",
        },
    ]

# Mostrar mensajes de chat desde el historial
for message in st.session_state["messages"]:
    if message["role"] == "system":
        continue
    elif message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="👨‍💻"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar="👤"):
            st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input():
    # Verificar si el contenido es inapropiado
    if check_for_inappropriate_content(prompt):
        with st.chat_message("assistant", avatar="👨‍💻"):
            st.markdown("Por favor, mantengamos la conversación respetuosa.")
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        output = generate_response(prompt)
        with st.chat_message("assistant", avatar="👨‍💻"):
            st.markdown(output)
