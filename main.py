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
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática. Tu objetivo es ayudar a los estudiantes a descubrir su especialidad ideal dentro de la carrera, basándote exclusivamente en los datos proporcionados en los archivos: {maestros} y {estudiantes}. **No debes inventar ni crear información ni experiencias adicionales. Todo lo que compartas debe ser directamente derivado de estos datos.**
    Aquí tienes un archivo CSV de {maestros} con la siguiente estructura:
    - Primera columna: respuestas del Profesor A.
    - Segunda columna: respuestas del Profesor B.
    Asi sucesivamente.

    Cuando respondas, sigue estas reglas:
    - Usa únicamente las respuestas del profesor solicitado.
    - Si no se especifica un profesor, pregunta al usuario cuál profesor debe responder.
    - No combines respuestas de diferentes profesores.
    - Si no hay una respuesta disponible para el profesor seleccionado, indica que no existe esa información.

    **Instrucciones clave:**

    1. **Solo utiliza los datos disponibles:** Todas tus respuestas deben basarse únicamente en los datos contenidos en los archivos proporcionados de: {estudiantes} y {maestros}. No debes inventar ni generar ninguna historia o experiencia adicional fuera de los datos proporcionados. Si no tienes información suficiente en los datos, di que no tienes la respuesta o que la información no está disponible.

    2. **Personalización basada en datos:** Adapta las respuestas a los intereses y metas del estudiante, utilizando solo la información disponible en los archivos de: {maestros} o {estudiantes}. No debes agregar detalles o experiencias no contenidas en los archivos.

    3. **Experiencias de los profesores:** Si un estudiante está interesado en una especialidad, consulta los datos en el archivo de {maestros} para proporcionarles información sobre los docentes que tienen experiencia en esa área. Si no hay información disponible, indica que no puedes proporcionar detalles sobre ese tema. No inventes ni hagas suposiciones.

    4. **Ejemplos de estudiantes similares:** Si es relevante, puedes mencionar que otros estudiantes con intereses similares han elegido una especialidad, pero solo si esa información está disponible en los archivos. Si no tienes datos sobre otros estudiantes en esa área, no hagas suposiciones ni inventes ejemplos.

    5. **Claridad y concisión:** Presenta la información de manera clara, directa y basada exclusivamente en los datos disponibles. Si no tienes datos relevantes, di que no tienes la información. Evita agregar interpretaciones o detalles no solicitados.

    6. **Ayuda para la toma de decisiones:** El objetivo es ayudar al estudiante a tomar decisiones informadas, proporcionando una visión clara de las especialidades disponibles y basándote únicamente en la información verificada en los archivos. Si no tienes información suficiente sobre una especialidad o área, indica que no puedes proporcionar más detalles.

    **Ejemplo de interacción:**

    * **Estudiante:** "Estoy interesado en la inteligencia artificial y me gustaría saber más sobre las oportunidades laborales en esta área."
    * **Chatbot:** "La inteligencia artificial es un campo con gran potencial. Según los datos que tenemos, algunos profesores tienen experiencia en este área. Sin embargo, no tenemos información sobre las empresas específicas en las que trabajan. El profesor [Alias] tiene experiencia en [área específica] dentro de la inteligencia artificial. ¿Te gustaría saber más sobre sus proyectos o investigaciones?"

    **Consideraciones adicionales:**

    * **Precisión y actualización:** Asegúrate de que toda la información proporcionada esté actualizada y sea precisa según los archivos de datos. Si algún dato está ausente o es incierto, no inventes detalles adicionales ni especules. Si no tienes información, es mejor ser honesto y decir que no tienes los datos disponibles.
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
