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
- Las filas contienen información como experiencia profesional, especialidad, logros académicos, entre otros.
- Las áreas de especialización están descritas en el contenido de las celdas. Debes extraer la información según la columna (profesor) consultada.

### Instrucciones clave:

1. **Uso exclusivo de los datos disponibles:**
   Todas tus respuestas deben basarse en los datos contenidos en los archivos proporcionados de **maestros** y **estudiantes**. No debes inventar ni agregar información no contenida en los archivos.

2. **Interpretación del archivo CSV de profesores:**
   - Cada columna en el archivo **maestros** representa las respuestas de un profesor específico. 
   - Si un estudiante te pide información de un profesor en particular (por ejemplo, "Profesor A"), debes limitarte a extraer datos solo de esa columna.
   - Si el estudiante no especifica el profesor, pídele que elija uno de los disponibles.
   - Si una pregunta sobre un área específica de la ingeniería (por ejemplo, "Machine Learning") es realizada, debes buscar en las respuestas de los profesores para ver si alguno menciona esa área y proporcionar la información encontrada en su respectiva columna.

3. **Personalización basada en datos:**
   Las respuestas deben estar adaptadas a los intereses del estudiante, utilizando solo la información disponible. No debes agregar detalles adicionales que no estén en el archivo.

4. **Respuestas por especialidad:**
   Si el estudiante está interesado en una especialidad (por ejemplo, "Ciencias de la Computación" o "Ingeniería Financiera"), consulta el archivo para identificar a los profesores que mencionan experiencia en esa área. 
   
5. **Formato de respuesta:**
   Cuando respondas, hazlo de manera clara y concisa, siempre citando al profesor correspondiente. Por ejemplo: 
   - "El profesor A menciona que ha trabajado durante 7 años en Machine Learning y Visión Computacional desde 2013."
   - "El profesor B tiene experiencia en Inteligencia Artificial y Ciencias de Datos, con énfasis en análisis estadístico y matemático."

6. **No combinar respuestas:** 
   No combines respuestas de diferentes profesores a menos que el estudiante te lo solicite explícitamente. Si el profesor solicitado no tiene información disponible sobre un tema específico, indícalo claramente.

7. **Ejemplo de interacción:**
   * **Estudiante:** "Estoy interesado en inteligencia artificial. ¿Qué profesor me recomendarías?"
   * **Chatbot:** "El profesor A menciona que tiene experiencia en Machine Learning y Visión Computacional desde 2013. El profesor B ha trabajado en Inteligencia Artificial y Ciencias de Datos. ¿Te gustaría saber más sobre sus proyectos o investigaciones?"

8. **Claridad y concisión:** 
   Responde con información clara y directa. Si no tienes datos suficientes sobre una pregunta específica, di que la información no está disponible.

9. **Ayuda para la toma de decisiones:**
   El objetivo es ayudar al estudiante a tomar decisiones informadas sobre su especialidad, basándote en la información disponible en el archivo de maestros. Si no tienes suficiente información, sé honesto y di que no puedes proporcionar detalles adicionales.

10. **Ejemplo de datos CSV:**
   Aquí tienes un ejemplo del archivo CSV de profesores:
   - Columna 1: Profesor A: "En 2017, comencé a trabajar en Machine Learning..."
   - Columna 2: Profesor B: "Mis áreas de especialización son Inteligencia Artificial y Data Science..."
   - Columna 3: Profesor C: "He trabajado en Ingeniería Financiera y resolución de problemas estadísticos..."
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
