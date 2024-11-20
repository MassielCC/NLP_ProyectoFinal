import pandas as pd
import streamlit as st
from datetime import datetime
from copy import deepcopy
from openai import OpenAI
import csv
import re
import pytz
import json
import logging

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar el cliente de OpenAI con la clave API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Configuración inicial de la página
st.set_page_config(page_title="SazónBot", page_icon=":pot_of_food:")
st.title("👨‍💻Nova-Infor")

# Mensaje de bienvenida
intro = """¡Bienvenido a Nova-Infor, tu consejero virtual"""
st.markdown(intro)

# Cargar desde un archivo CSV
def load(file_path):
    """Cargar el menú desde un archivo CSV con columnas Plato, Descripción y Precio."""
    load = pd.read_csv(file_path)
    return load

# Cargar 
maestros = load("Entrevistas_maestros.csv")
estudiantes = load("Entrevistas_estudiantes.csv")

def get_system_prompt(maestros, estudiantes):
    """Define el prompt del sistema para el bot de orientación académica con respuestas cortas y basadas en información real."""
    system_prompt = f"""
    Eres un chatbot de orientación académica para estudiantes de Ingeniería Informática. Tu función es ayudar a los estudiantes a elegir una especialidad basada en sus intereses y habilidades, usando la información de los archivos {maestros} y {estudiantes}. No debes inventar historias ni agregar información que no esté en estos archivos.

**Base de conocimiento:**

* **Expertos:** Tienes acceso a entrevistas y respuestas de especialistas en diversas áreas de Ingeniería Informática contenidas en {maestros}. Solo debes usar esta información para describir las especialidades.
* **Estudiantes:** También tienes acceso a testimonios reales de estudiantes en {estudiantes}, quienes han compartido sus experiencias al elegir su especialidad. Usa esta información si el estudiante está indeciso o necesita ejemplos reales.

**Tareas del chatbot:**

1. **Saludo breve:** Saluda al estudiante y explica tu función.
2. **Preguntas rápidas:** Haz preguntas directas para entender los intereses del estudiante, como "¿Qué áreas te interesan más: inteligencia artificial, desarrollo web o ciberseguridad?".
3. **Recomendaciones cortas:** Basado en los intereses del estudiante, ofrece recomendaciones breves y concretas de especialidades, utilizando solo la información de {maestros}.
4. **Experiencias de estudiantes:** Si el estudiante está indeciso, menciona ejemplos cortos y reales de otros estudiantes de {estudiantes} que pasaron por la misma situación, sin inventar detalles adicionales.
5. **Comparación breve:** Si es necesario, compara especialidades con descripciones simples, destacando solo las diferencias principales.
6. **Respuestas claras:** Responde preguntas frecuentes de manera concisa y utilizando solo la información de los archivos {maestros} y {estudiantes}.
7. **Cierre rápido:** Ofrece un resumen final y agradece al estudiante por su tiempo.

**Ejemplo de interacción:**

1. "Hola, soy tu asistente para elegir una especialidad en Ingeniería Informática. ¿Qué áreas de la informática te interesan más?"
2. "Basado en lo que mencionas, Inteligencia Artificial podría ser una buena opción. Según los profesores en {maestros}, es un área con alta demanda y uso intensivo de matemáticas."
3. "Si no estás seguro, no te preocupes. Un estudiante en {estudiantes} también dudaba entre IA y desarrollo web. Él exploró ambas áreas y finalmente eligió IA por su pasión por los datos."

**Instrucciones clave:**

- Usa **solo** la información de los archivos {maestros} y {estudiantes}.
- Responde de forma breve y precisa.
- No inventes ni añadas información que no esté en los archivos.
    """
    return system_prompt.replace("\n", " ")


def generate_response(prompt, temperature=0, max_tokens=1000):
    """Enviar el prompt a OpenAI y devolver la respuesta con un límite de tokens."""
    st.session_state["messages"].append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state["messages"],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    response = completion.choices[0].message.content
    st.session_state["messages"].append({"role": "assistant", "content": response})
    return response

# Función para verificar contenido inapropiado
def check_for_inappropriate_content(prompt):
    """Verifica si el prompt contiene contenido inapropiado utilizando la API de Moderación de OpenAI."""
    try:
        response = client.moderations.create(input=prompt)
        logging.info(f"Moderation API response: {response}")
        moderation_result = response.results[0]
        # Verifica si está marcado como inapropiado
        if moderation_result.flagged:
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error al llamar a la API de Moderación: {e}")
        return False

# Ajustar el tono del bot
def adjust_tone(tone="friendly"):
    """Ajustar el tono del bot según las preferencias del cliente."""
    if tone == "formal":
        st.session_state["tone"] = "formal"
        return "Eres un asistente formal y educado."
    else:
        st.session_state["tone"] = "friendly"
        return "Eres un asistente amigable y relajado."

# Estado inicial de la conversación
initial_state = [
    {"role": "system", "content": get_system_prompt(maestros, estudiantes)},
    {
        "role": "assistant",
        "content": f"¡Hola! Soy tu asistente virtual para elegir la especialidad ideal en Ingeniería Informática. Para comenzar, cuéntame un poco sobre ti.",
    },
]

if "messages" not in st.session_state:
    st.session_state["messages"] = deepcopy(initial_state)

# Botón para eliminar conversación
clear_button = st.button("Eliminar conversación", key="clear")
if clear_button:
    st.session_state["messages"] = deepcopy(initial_state)

# Mostrar mensajes de chat desde el historial al recargar la aplicación
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
