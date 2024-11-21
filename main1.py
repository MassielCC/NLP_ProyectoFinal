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
from fuzzywuzzy import fuzz

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

# Cargar desde un archivo CSV
def load(file_path):
    """Cargar datos desde un archivo CSV."""
    data = pd.read_csv(file_path)
    return data

# Cargar los datos
maestros_df = load("Entrevistas_maestros.csv")
estudiantes_df = load("Entrevistas_estudiantes.csv")

# Crear un diccionario con las respuestas de los maestros
maestros_data = {}
for index, row in maestros_df.iterrows():
    pregunta = row['Pregunta']
    maestros_data[pregunta] = {
        'Profesor A': row.get('Profesor A', ''),
        'Profesor B': row.get('Profesor B', '')
    }

# Crear un diccionario con las respuestas de los estudiantes (si es necesario)
estudiantes_data = {}
for index, row in estudiantes_df.iterrows():
    pregunta = row['Pregunta']
    estudiantes_data[pregunta] = {
        'Estudiante A': row.get('Estudiante A', ''),
        'Estudiante B': row.get('Estudiante B', '')
    }

def get_system_prompt():
    """Define el prompt del sistema para el chatbot."""
    system_prompt = """
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática.
    Tu objetivo es ayudar a los estudiantes a descubrir su especialidad ideal dentro de la carrera.
    Si no tienes una respuesta directa en tus datos, proporciona una respuesta general y útil.
    """
    return system_prompt

def buscar_respuesta(pregunta_usuario):
    """Buscar una respuesta en los datos de los maestros."""
    max_similarity = 0
    best_match = None
    for pregunta in maestros_data.keys():
        similarity = fuzz.ratio(pregunta.lower(), pregunta_usuario.lower())
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = pregunta
    if max_similarity > 70:  # Umbral de similitud
        return maestros_data[best_match]
    else:
        return None

def generate_response(prompt, temperature=0.1, max_tokens=1000):
    """Generar una respuesta basada en los datos o utilizando OpenAI."""
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Intentar encontrar una respuesta en los datos
    respuesta_datos = buscar_respuesta(prompt)
    
    if respuesta_datos:
        # Si se encuentra una respuesta, devolverla
        response = ""
        if respuesta_datos['Profesor A']:
            response += f"**Profesor A dice:** {respuesta_datos['Profesor A']}\n\n"
        if respuesta_datos['Profesor B']:
            response += f"**Profesor B dice:** {respuesta_datos['Profesor B']}\n"
        st.session_state["messages"].append({"role": "assistant", "content": response})
        return response
    else:
        # Si no se encuentra respuesta, usar OpenAI
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state["messages"],
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
initial_state = [
    {"role": "system", "content": get_system_prompt()},
    {
        "role": "assistant",
        "content": "¡Hola! Soy tu asistente virtual para elegir la especialidad ideal en Ingeniería Informática. Para comenzar, cuéntame un poco sobre ti.",
    },
]

if "messages" not in st.session_state:
    st.session_state["messages"] = deepcopy(initial_state)

# Botón para eliminar conversación
clear_button = st.button("Eliminar conversación", key="clear")
if clear_button:
    st.session_state["messages"] = deepcopy(initial_state)

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

