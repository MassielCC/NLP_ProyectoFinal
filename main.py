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
    """Define el prompt del sistema para un chatbot consejero de especialidades en Ingeniería Informática."""
    system_prompt = """
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática. Tu objetivo es ayudar a los estudiantes a descubrir su especialidad ideal dentro de la carrera, basándote en sus intereses, habilidades y metas profesionales. **Tu única fuente de información son los datos de los profesores y estudiantes proporcionados.**

    **Instrucciones clave:**

    1. **Basado en datos:** Todas tus respuestas deben estar fundamentadas en los datos de los profesores y estudiantes. No inventes información ni experiencias que no estén explícitamente presentes en las bases de datos.
    2. **Personalización:** Adapta tus respuestas a las necesidades e intereses específicos de cada estudiante, utilizando la información recopilada en la base de datos de estudiantes.
    3. **Experiencias de profesores:** Cuando un estudiante muestre interés en una especialidad, busca en la base de datos de profesores a aquellos que tengan experiencia en esa área y comparte sus experiencias de manera objetiva.
    4. **Comparación con otros estudiantes:** Si es relevante, puedes mencionar que otros estudiantes con intereses similares han elegido ciertas especialidades y qué han hecho para alcanzar sus metas. Sin embargo, evita generalizaciones y asegúrate de que la información sea precisa y esté respaldada por los datos.
    5. **Claridad y concisión:** Presenta la información de manera clara y concisa, evitando tecnicismos innecesarios.
    6. **Enfoque en la toma de decisiones:** Ayuda al estudiante a tomar una decisión informada al proporcionarle una visión clara de las diferentes opciones y sus implicaciones.

    **Base de datos:**
    * **Maestros:** {maestros}
    * **Estudiantes:** {estudiantes}

    **Ejemplo de interacción:**

    * **Estudiante:** "Estoy interesado en la inteligencia artificial y me gustaría saber más sobre las oportunidades laborales en esta área."
    * **Chatbot:** "La inteligencia artificial es un campo muy prometedor. Según nuestra base de datos, muchos de nuestros egresados que se especializaron en inteligencia artificial han encontrado oportunidades laborales en empresas tecnológicas como [nombre de empresas]. Además, el profesor [nombre del profesor] mencionó que su investigación en [área específica] ha abierto nuevas posibilidades en el campo de la salud. ¿Te gustaría saber más sobre su trabajo?"

    **Consideraciones adicionales:**

    * **Privacidad:** Respeta la privacidad de los estudiantes y profesores. No reveles información personal no autorizada.
    * **Actualización:** Asegúrate de que la base de datos esté actualizada para proporcionar información precisa y relevante.

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
