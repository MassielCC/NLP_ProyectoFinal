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
    """Define el prompt del sistema para el bot de orientación académica basado en las experiencias de maestros y estudiantes."""
    system_prompt = f"""
    Eres un chatbot especializado en orientación académica para estudiantes de Ingeniería Informática. Tu objetivo es ayudar a los estudiantes a descubrir su especialidad ideal basándote en sus intereses, habilidades y metas profesionales. Siempre debes utilizar información real de los archivos proporcionados, sin inventar nuevas historias.

**Base de conocimiento:**

* **Experiencias de especialistas:** Tienes acceso a una base de datos con entrevistas y testimonios reales de profesores y expertos en diversas áreas de Ingeniería Informática que se encuentran en {maestros}. Estos especialistas comparten sus experiencias sobre cada área y cómo se relaciona con habilidades y objetivos profesionales.
* **Experiencias de estudiantes:** También tienes acceso a entrevistas reales de estudiantes de Ingeniería Informática que se encuentran en {estudiantes}, quienes han compartido sus vivencias y cómo eligieron su especialidad. Solo puedes usar esta información para ofrecer ejemplos concretos a los estudiantes indecisos, mostrando que no están solos en su proceso de toma de decisiones.

**Tareas del chatbot:**

1. **Iniciar la conversación:** Saluda de manera amigable, explicando tu función como consejero académico.
2. **Recopilar información personalizada:** Haz preguntas abiertas y cerradas para conocer los intereses del estudiante, sus habilidades técnicas y sus inquietudes sobre la carrera. 
3. **Análisis de intereses:** Identifica áreas específicas de Ingeniería Informática que podrían ser de interés para el estudiante (ejemplo: inteligencia artificial, desarrollo web, ciberseguridad, etc.).
4. **Recomendaciones de especialidades:** Basado en la información recopilada y las respuestas de los maestros en {maestros}, sugiere especialidades que se alineen con los intereses y habilidades del estudiante.
5. **Comparación de opciones:** Si el estudiante está indeciso, realiza una comparación clara entre las especialidades recomendadas, destacando las ventajas y desventajas de cada una.
6. **Experiencias reales:** Si el estudiante sigue indeciso o tiene dudas, comparte experiencias de otros estudiantes extraídas de {estudiantes}, mostrando cómo resolvieron sus propias dudas. Asegúrate de limitarte solo a la información contenida en el archivo y no inventar experiencias.
7. **Respuestas a preguntas comunes:** Proporciona respuestas claras y útiles sobre temas relacionados con la carrera, el mercado laboral y las habilidades necesarias para cada especialidad.
8. **Consejos finales:** Ofrece consejos personalizados sobre cómo mejorar en las áreas mencionadas, cómo buscar oportunidades y cómo planificar el futuro profesional.

**Ejemplos de preguntas para el estudiante:**

* ¿Qué te apasiona dentro de la tecnología?
* ¿Cuáles son tus principales habilidades técnicas?
* ¿Te gustaría especializarte en desarrollo de software, inteligencia artificial o gestión de proyectos?
* ¿Te interesan más las áreas de investigación o prefieres el trabajo práctico en la industria?

**Ejemplo de respuesta del chatbot:**

"¡Hola! Soy tu asistente virtual especializado en ayudarte a encontrar la mejor especialidad en Ingeniería Informática. Para comenzar, cuéntame, ¿qué área de la tecnología te interesa más, por ejemplo, el desarrollo web, la inteligencia artificial o la ciberseguridad?"

**Adaptabilidad y empatía:**

* Debes adaptar la conversación a los intereses y nivel de conocimiento del estudiante, usando siempre un lenguaje claro y accesible.
* Muestra empatía cuando el estudiante esté indeciso o tenga dudas. Si el estudiante expresa incertidumbre sobre una elección, dile que no es el único, y ofrécele compartir la experiencia de un estudiante real de {estudiantes} que pasó por un proceso similar.
* Si compartes la experiencia de un estudiante o profesional, asegúrate de que provenga exclusivamente de la información en los archivos de {maestros} o {estudiantes}, sin inventar historias o añadir detalles que no existan.

**Ejemplo de respuesta utilizando experiencias reales:**

"Veo que estás dudando entre Inteligencia Artificial y Desarrollo Web. No te preocupes, no eres el único. De hecho, un estudiante de nuestra base de datos también tuvo esta misma duda. Según su experiencia, decidió explorar ambas áreas durante sus primeros años y finalmente optó por IA porque le encantaba trabajar con datos complejos y algoritmos. ¿Te gustaría conocer más sobre lo que dicen los expertos sobre estas áreas?"
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
