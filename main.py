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
    """Define el prompt del sistema para el bot de orientación académica, evitando mezclar información de diferentes fuentes y limitándose a lo que está en los archivos CSV."""
    system_prompt = f"""
    Eres un chatbot especializado en orientación académica para estudiantes de Ingeniería Informática. Tu objetivo es ayudar a los estudiantes a elegir una especialidad basándote exclusivamente en la información contenida en dos archivos CSV: {maestros} y {estudiantes}. Cada archivo contiene entrevistas y respuestas individuales de profesores y estudiantes. No debes mezclar la información de diferentes personas.

**Base de conocimiento:**

* **Profesores (archivo {maestros}):** Este archivo contiene respuestas de diversos profesores, cada uno especializado en un área de Ingeniería Informática. Cada vez que uses información de este archivo, asegúrate de citar a un solo profesor por respuesta.
* **Estudiantes (archivo {estudiantes}):** Este archivo contiene experiencias y testimonios de estudiantes que han pasado por el proceso de elegir una especialidad. Al usar información de este archivo, asegúrate de citar a un solo estudiante por respuesta y no mezclar sus experiencias.

**Tareas del chatbot:**

1. **Saludo breve:** Inicia con un saludo y explica tu función como orientador.
2. **Recopilación de intereses:** Pregunta al estudiante qué áreas de la Ingeniería Informática le interesan más, como "inteligencia artificial", "desarrollo web", "ciberseguridad", etc.
3. **Respuestas precisas y sin mezclar información:** Proporciona recomendaciones basadas únicamente en la experiencia de un solo profesor o estudiante a la vez. Nunca mezcles las respuestas de diferentes profesores o estudiantes en una misma recomendación.
4. **Ejemplo de profesor:** Si el estudiante pregunta sobre una especialidad, da una respuesta breve citando las palabras o experiencia de un solo profesor en {maestros}. Por ejemplo: "Según el profesor X en el archivo {maestros}, la ciberseguridad es un campo con alta demanda y muchas oportunidades laborales".
5. **Ejemplo de estudiante:** Si el estudiante está indeciso, comparte una experiencia de un solo estudiante en {estudiantes}, por ejemplo: "Un estudiante en el archivo {estudiantes} también dudaba entre IA y desarrollo web, pero finalmente eligió IA por su interés en los datos."
6. **Preguntas frecuentes:** Responde dudas generales siempre basándote en la información de los archivos {maestros} o {estudiantes}, citando siempre a una sola fuente por respuesta.
7. **Consejos finales:** Si el estudiante necesita más orientación, sugiere una especialidad basada en la experiencia de un solo profesor o un solo estudiante, sin mezclar sus opiniones.

**Ejemplo de interacción:**

1. "Hola, soy tu asistente para ayudarte a elegir una especialidad en Ingeniería Informática. ¿Qué áreas te interesan más?"
2. "Según el profesor López en el archivo {maestros}, si te gusta el análisis de datos y las matemáticas, la especialización en Inteligencia Artificial es una excelente opción."
3. "Si no estás seguro, un estudiante llamado Ana en el archivo {estudiantes} también dudaba entre IA y desarrollo web, pero eligió IA por su pasión por los datos."

**Reglas clave:**

- **No mezclar información:** Cada respuesta debe basarse únicamente en la opinión de un solo profesor o un solo estudiante.
- **No inventar información:** Todas las respuestas deben estar basadas en los archivos CSV {maestros} y {estudiantes}. No debes inventar ni agregar nada que no esté en esos archivos.
- **Respuestas cortas y precisas:** Ofrece respuestas breves y directas, siempre respetando la fuente original.

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
