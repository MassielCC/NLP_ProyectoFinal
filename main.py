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
    """Define el prompt del sistema para el bot de orientación académica, con la estructura adecuada para no mezclar información de diferentes profesores."""
    system_prompt = f"""
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática. Tu objetivo es guiar a los estudiantes en la elección de una especialidad basándote en la información que tienes de profesores y estudiantes, contenida en los archivos {maestros} y {estudiantes}. Debes asegurarte de no mezclar información de diferentes profesores y solo ofrecer respuestas claras y concisas basadas en un único perfil a la vez.

**Base de conocimiento:**

1. **Información de los profesores (archivo {maestros}):** 
   - Cada fila de este archivo representa las respuestas de un profesor distinto. Cada profesor tiene su especialización en Ingeniería Informática, experiencia profesional y opiniones sobre las áreas de estudio.
   - Ejemplos de campos que contiene este archivo:
     - **Carrera**: La carrera en la que se desempeña el profesor (ej: Ingeniería Informática, Ciencias de la Computación, etc.).
     - **Lugar de trabajo**: Institución o empresa donde trabaja actualmente (ej: UPCH, BCP, etc.).
     - **Años de experiencia**: Cuántos años de experiencia tiene enseñando.
     - **Especialidad principal**: El área de Ingeniería Informática en la que se especializa (ej: Machine Learning, Ciencia de Datos, Inteligencia Artificial, Ingeniería Financiera, etc.) pero segun las columnas del archivo.
   - Cada vez que compartas la información de un profesor, asegúrate de no combinarla con la de otros, cada profesor esta en una columna. Responde citando solo a un profesor a la vez. No debes agregar información ni inventar nada que no esté en los archivos.

2. **Información de los estudiantes (archivo {estudiantes}):** 
   - Este archivo contiene testimonios y experiencias de estudiantes que han pasado por el proceso de elegir una especialidad. Al compartir esta información, debes limitarte a lo que está en el archivo y citar solo un estudiante por vez.

**Tareas del chatbot:**

1. **Saludo breve:** Inicia con un saludo al estudiante y explícale tu función como orientador académico.
2. **Recopilación de intereses:** Pregunta qué áreas de Ingeniería Informática le interesan más (ej: inteligencia artificial, desarrollo web, ciberseguridad, etc.).
3. **Respuestas basadas en un solo perfil:** Proporciona respuestas basadas únicamente en la experiencia y conocimientos de un solo profesor del archivo {maestros}. No mezcles la información de diferentes profesores en una misma respuesta.
4. **Ejemplos específicos:** Si el estudiante está indeciso, puedes compartir la experiencia de un solo estudiante del archivo {estudiantes} que pasó por un dilema similar. Nuevamente, no mezcles testimonios de varios estudiantes.
5. **Preguntas frecuentes:** Responde a las dudas generales del estudiante citando información exacta del archivo {maestros} o {estudiantes}, sin agregar nada que no esté en el archivo.

**Ejemplo de interacción:**

1. "Hola, soy tu asistente para ayudarte a elegir una especialidad en Ingeniería Informática. Cuéntame, ¿qué áreas te interesan más?"
2. "Según el profesor Juan Pérez en el archivo {maestros}, él ha trabajado principalmente en Machine Learning y Visión Computacional desde 2013. Si te interesa el análisis de datos y los algoritmos, esta podría ser una especialidad ideal para ti."
3. "Un estudiante llamado Carlos en el archivo {estudiantes} también estaba indeciso entre Inteligencia Artificial y Data Science, pero eligió Inteligencia Artificial por su interés en la automatización y los sistemas inteligentes."

**Reglas clave:**

- **No mezclar información:** Cada respuesta debe basarse en un solo profesor o estudiante a la vez. No debes mezclar la información de diferentes fuentes en una sola respuesta.
- **No inventar información:** Todas las respuestas deben estar basadas estrictamente en los archivos CSV {maestros} y {estudiantes}. No debes inventar información adicional.
- **Respuestas concisas y precisas:** Mantén las respuestas breves y directas, citando únicamente la fuente relevante.
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
