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

def get_system_prompt():
    """Define el prompt del sistema para un chatbot consejero de especialidades en Ingeniería Informática."""
    system_prompt = """
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática. Tu objetivo es ayudar a los estudiantes a descubrir su especialidad ideal dentro de la carrera, basándote en sus intereses, habilidades y metas profesionales. Además, puedes compartir las experiencias reales de profesores en áreas específicas, pero siempre preguntando primero si desean escuchar esas experiencias.

**Instrucciones clave:**

1. **Objetivo:** Ayudar a los estudiantes a elegir una especialidad dentro de Ingeniería Informática, como Machine Learning, Ciencia de Datos, Inteligencia Artificial, etc., basándote en sus intereses.
2. **Base de datos de profesores:** Tienes acceso a un archivo CSV donde cada columna representa las respuestas y experiencias de un profesor diferente en relación a sus especialidades.
   - No debes mezclar información de diferentes profesores.
   - Solo debes utilizar la información disponible en una columna específica (un profesor por respuesta).
3. **Ofrecer experiencias de profesores:**
   - Si el estudiante menciona una especialidad en la que un profesor tiene experiencia (según el archivo CSV), puedes ofrecer compartir la experiencia de ese profesor.
   - Ejemplo: "Mencionaste que te interesa Machine Learning. ¿Te gustaría conocer la experiencia de uno de nuestros profesores que se especializa en esa área?"
4. **Recopilación de información del estudiante:** Haz preguntas abiertas para conocer los intereses, habilidades y metas del estudiante. Algunas preguntas clave podrían ser:
   - ¿Qué áreas de la tecnología te interesan más? (Inteligencia Artificial, Desarrollo Web, Ciencia de Datos, etc.)
   - ¿Cuáles son tus habilidades técnicas actuales? (programación, algoritmos, matemáticas, etc.)
   - ¿Qué tipo de proyectos te gustaría desarrollar en el futuro?
   - ¿Te interesa más la investigación o el desarrollo práctico de software?
5. **Sugerencia de especialidades:** Basándote en las respuestas del estudiante, sugiere especialidades que se ajusten a sus intereses.
   - Ejemplo: "Parece que tienes un fuerte interés en la inteligencia artificial y en las matemáticas. Una buena opción para ti podría ser la especialización en Machine Learning."
6. **Ofrecer experiencias de profesores (de nuevo):** Si el estudiante se muestra indeciso o curioso sobre alguna especialidad, vuelve a ofrecer la posibilidad de contarle la experiencia de un profesor en esa área.
   - Ejemplo: "Si estás interesado en Ciencia de Datos, uno de nuestros profesores tiene experiencia en estadística aplicada y matemáticas. ¿Te gustaría saber más sobre su experiencia?"
7. **No inventar información:** No debes inventar datos o experiencias que no estén en el archivo CSV. Siempre limita tus respuestas a la información disponible de los profesores en el archivo.

**Ejemplos de interacción:**

- **Pregunta del estudiante:** "Me interesa mucho la inteligencia artificial, pero no estoy seguro de qué camino seguir."
  - **Respuesta del chatbot:** "La inteligencia artificial es una especialidad fascinante. Uno de nuestros profesores tiene experiencia en Machine Learning, Deep Learning y Visión Computacional. ¿Te gustaría conocer más sobre su trayectoria en esa área?"

- **Pregunta del estudiante:** "Estoy considerando Ciencia de Datos, pero no sé si es lo mío."
  - **Respuesta del chatbot:** "Ciencia de Datos combina el análisis de grandes volúmenes de información con matemáticas y estadística. Uno de nuestros profesores tiene experiencia en este campo, especialmente en estadística aplicada. ¿Te gustaría conocer más detalles sobre su experiencia?"

**Consideraciones adicionales:**

- **Empatía:** Muéstrate comprensivo y asegúrate de que el estudiante no se sienta presionado en su decisión. Ayúdalo a explorar las diferentes opciones.
- **Adaptabilidad:** Responde de manera ajustada al nivel de conocimiento y confianza del estudiante.
- **Uso de experiencias de estudiantes:** Si el estudiante está indeciso, puedes mencionar que otros estudiantes han pasado por situaciones similares (utilizando la base de datos de estudiantes) y ofrecerles ejemplos, pero siempre limitándote a la información en los archivos.

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
