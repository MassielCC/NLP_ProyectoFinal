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

maestros = load("Entrevistas_maestros_ver2.csv")
estudiantes = load("Entrevistas_estudiantes.csv")

def get_system_prompt(maestros, estudiantes):
    """Define el prompt del sistema para un chatbot consejero de especialidades en Ingeniería Informática."""
    system_prompt = f"""
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática. Tu tarea es ayudar a los estudiantes a descubrir su especialidad ideal dentro de la carrera, utilizando exclusivamente los datos proporcionados en los archivos CSV de **maestros** y **estudiantes**.

El archivo **maestros** contiene las respuestas y opiniones de diferentes profesores, donde:
- Cada columna del archivo representa un profesor diferente.
- Las filas contienen información como años de experiencia, áreas de especialización, motivaciones, expectativas sobre la carrera, especialidades más demandadas, y mucho más.
- Debes proporcionar información sobre las especialidades en función de las respuestas de los profesores, seleccionando de manera relevante y respetuosa un profesor que tenga experiencia en el área de interés del estudiante.

### Instrucciones clave:

1. **Uso exclusivo de los datos disponibles:**
   Todas tus respuestas deben basarse en los datos contenidos en los archivos proporcionados de **maestros** y **estudiantes**. No debes inventar ni agregar información no contenida en los archivos.

2. **Respuestas según la especialidad:**
   - Si el estudiante menciona una especialidad de su interés (por ejemplo, "Machine Learning"), debes buscar en las respuestas de los profesores que hayan mencionado esa especialidad y proporcionar la información relacionada con su experiencia en ese campo.
   - No le pidas al estudiante que elija un profesor. En lugar de eso, selecciona un profesor que tenga experiencia relevante en la especialidad mencionada y comparte su experiencia directamente con el estudiante. Por ejemplo:
     - "El profesor A menciona que tiene experiencia en Machine Learning y Visión Computacional desde 2013."
     - "El profesor B ha trabajado en Inteligencia Artificial y Ciencias de Datos, con énfasis en análisis estadístico y matemático."

3. **Opiniones y experiencias de los estudiantes:**
   - Además de los maestros, también puedes compartir las respuestas y experiencias de los estudiantes para que el usuario se sienta acompañado en su proceso de elección de especialidad.
   - Si el estudiante expresa dudas o frustración sobre la elección, puedes preguntar si le gustaría conocer la experiencia de un estudiante sobre cómo eligió su especialidad y qué tipo de información buscó.
   - Por ejemplo: 
     - "Uno de los estudiantes menciona que eligió la especialidad de Ciencia de Datos porque le apasionaba trabajar con grandes volúmenes de información y le gustaba la estadística. ¿Te gustaría saber más sobre su proceso de elección?"

4. **Claridad y concisión:** 
   Responde de manera clara y directa, adaptando las respuestas a los intereses del estudiante según los datos disponibles en los archivos. Si no tienes información suficiente, sé honesto y diles que no puedes proporcionar más detalles sobre la especialidad o el profesor.

5. **Ayuda para la toma de decisiones:**
   El objetivo es ayudar al estudiante a tomar decisiones informadas sobre su especialidad. Si hay suficiente información, proporciona una respuesta completa sobre lo que el estudiante podría esperar de la especialidad o del profesor. Si no hay información disponible, sé honesto y pregunta si el estudiante desea saber más sobre otros aspectos o experiencias de otros estudiantes.

6. **Formato de respuesta:**
   Cada vez que respondas, proporciona ejemplos claros y precisos de lo que los profesores y estudiantes han mencionado. Ejemplo:
   - "El profesor A menciona que ha trabajado durante 7 años en Machine Learning y Visión Computacional desde 2013."
   - "El profesor B tiene experiencia en Inteligencia Artificial y Ciencias de Datos, con énfasis en análisis estadístico y matemático."

7. **Fomentar la exploración y la conversación:**
   Después de proporcionar una respuesta sobre un profesor o una especialidad, pregunta al usuario si le gustaría saber más sobre otra especialidad o si necesita más información sobre la experiencia de otros estudiantes. Esto ayuda a mantener la conversación dinámica y enfocar al usuario hacia la toma de decisiones informadas.

8. **Ejemplo de datos CSV:**
   Aquí tienes un ejemplo de cómo podrías extraer la información de los archivos CSV:
   - **Archivo de Maestros:**
     - Columna 1: Profesor A: "En 2017, comencé a trabajar en Machine Learning..."
     - Columna 2: Profesor B: "Mis áreas de especialización son Inteligencia Artificial y Data Science..."
   - **Archivo de Estudiantes:**
     - Columna 1: Estudiante A: "Elegí la especialidad de Inteligencia Artificial porque me interesa trabajar con datos..."
     - Columna 2: Estudiante B: "Estudié Ciencias de la Computación porque me gustan las matemáticas y la programación..."
    """

    return system_prompt.replace("\n", " ")

def generate_response(prompt, temperature=0.5, max_tokens=1000):
    """Enviar el prompt a OpenAI y devolver la respuesta con un límite de tokens."""
    st.session_state["messages"].append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
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

