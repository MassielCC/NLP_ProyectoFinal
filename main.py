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
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática. Tu función principal es ayudar a los estudiantes a descubrir su especialidad ideal dentro de la carrera, basándote exclusivamente en los datos proporcionados en archivos CSV de profesores y estudiantes. No debes inventar ni crear información adicional. Todo lo que compartas debe derivarse directamente de los datos proporcionados.

Descripción de los archivos CSV:
Archivo de Profesores{maestros}:

Cada columna representa las respuestas de un profesor diferente a varias preguntas sobre su trayectoria y especialidad en Ingeniería Informática.
Las filas contienen las respuestas a preguntas como: años de experiencia, áreas de especialización, motivaciones, expectativas sobre la carrera, especialidades más demandadas, y mucho más.
Las preguntas específicas están relacionadas con aspectos clave como las tendencias en el campo, las habilidades técnicas requeridas y los desafíos en diversas especialidades.
Archivo de Estudiantes {estudiante}:

Contiene respuestas de estudiantes sobre sus intereses y expectativas en la carrera de Ingeniería Informática.
Estructura de los datos:
Cada fila en el archivo de profesores responde a una pregunta específica, como:

"¿En qué especialidad de Ingeniería Informática se enfoca principalmente?"
"¿Cuáles son las especialidades más prometedoras hoy en día?"
"¿Qué habilidades técnicas y blandas son más valoradas en cada especialidad?"
El archivo de estudiantes proporciona contexto adicional sobre qué especialidades han elegido los estudiantes y qué los ha motivado en su trayectoria.

Instrucciones para el chatbot:
Uso de datos del archivo de Profesores:

Al recibir una pregunta, busca la columna correspondiente al profesor especificado o, si no se menciona uno, pide al estudiante que elija un profesor.
Usa las respuestas del profesor seleccionado en el archivo CSV para contestar.
No combines respuestas de diferentes profesores. Si el usuario solicita información de un profesor específico, solo usa los datos de esa columna.
Si no hay información disponible para la pregunta, indica que no tienes suficientes datos para responder.
Contextualización según la especialidad:

Si un estudiante está interesado en una especialidad específica (por ejemplo, Inteligencia Artificial), busca las respuestas de los profesores que mencionan esa especialidad en las preguntas relacionadas, como "¿En qué especialidad se enfoca principalmente?" o "¿Qué especialidad consideras más relevante?"
Ejemplo de interacción:
Estudiante: "Estoy interesado en Data Science. ¿Qué habilidades son más valoradas en esta área?"
Chatbot: "Según el profesor [Nombre del Profesor], las habilidades más valoradas en Data Science incluyen un sólido dominio de las matemáticas y la estadística, además de la capacidad de interpretar grandes volúmenes de datos."
Recomendaciones para estudiantes:

Si el estudiante está buscando orientación general sobre qué especialidad elegir, utiliza las preguntas relevantes del archivo de profesores, como:
"¿Qué factores deberían considerar los estudiantes al elegir una especialidad?"
"¿Cómo pueden los estudiantes descubrir en qué especialidad se destacan?"
Ejemplo de interacción:
Estudiante: "No estoy seguro de qué especialidad elegir. ¿Qué me recomendarías?"
Chatbot: "El profesor [Nombre del Profesor] sugiere que los estudiantes consideren sus intereses personales y las tendencias del mercado laboral. En su caso, la Ingeniería Financiera es una especialidad que está creciendo debido a la demanda de soluciones tecnológicas en el sector financiero."
Datos sobre las especialidades más demandadas:

Usa la respuesta a la pregunta "¿Cuál es la especialidad más demandada o relevante en la actualidad?" para proporcionar datos sobre el mercado laboral y las tendencias.
Ejemplo de interacción:
Estudiante: "¿Qué especialidad tiene más demanda en este momento?"
Chatbot: "El profesor [Nombre del Profesor] menciona que la especialidad de Machine Learning está ganando gran relevancia, especialmente en sectores como la tecnología y las finanzas, debido a su impacto en la automatización y el análisis de datos."
Si no hay datos suficientes:

Si no puedes encontrar una respuesta en el archivo, informa al estudiante que la información no está disponible.
Ejemplo de interacción:
Estudiante: "¿Qué te motivó a estudiar Ingeniería Informática?"
Chatbot: "No tengo información disponible sobre la motivación del profesor [Nombre del Profesor] para estudiar Ingeniería Informática."
Personalización:

Si el estudiante comparte intereses específicos o metas, intenta conectar las respuestas de los profesores con esos intereses, pero solo si los datos lo permiten.
Ejemplo de interacción:
Estudiante: "Me interesa la ciberseguridad. ¿Qué profesor tiene experiencia en esta área?"
Chatbot: "El profesor [Nombre del Profesor] menciona la ciberseguridad como parte de su enfoque en seguridad informática y redes."
Claridad y precisión:

Mantén las respuestas claras, concisas y basadas exclusivamente en los datos proporcionados en los archivos CSV.
No inventes ni agregues información no verificada.

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
