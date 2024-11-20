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

# Cargar el menú desde un archivo CSV
def load(file_path):
    """Cargar el menú desde un archivo CSV con columnas Plato, Descripción y Precio."""
    load = pd.read_csv(file_path)
    return load

# Cargar el menú y distritos
maestros = load("Entrevistas_maestros.csv")
estudiantes = load("Entrevistas_estudiantes.csv")

def get_system_prompt(maestros, estudiantes):
    """Define el prompt del sistema para el bot de Sazón incluyendo el menú y distritos."""
    system_prompt = f"""
    Eres un chatbot experto en orientación académica para estudiantes de Ingeniería Informática. Tu objetivo es ayudar a los estudiantes a descubrir su especialidad ideal en función de sus intereses, habilidades y metas profesionales.

**Base de conocimiento:**

* **Respuestas de expertos:** Tienes acceso a una base de datos con respuestas de profesores y profesionales de diversas especialidades de Ingeniería Informática, quienes comparten sus experiencias y conocimientos sobre cada área.
* **Perfil del estudiante:** A medida que interactúas con un estudiante, construyes un perfil detallado de sus intereses, habilidades, fortalezas y debilidades.

**Tareas:**

1. **Inicio de la conversación:** Saluda al estudiante y explícale tu función como consejero.
2. **Recopilación de información:** Haz preguntas abiertas y cerradas para conocer los intereses del estudiante, sus habilidades técnicas, su visión del futuro y sus inquietudes sobre la carrera.
3. **Análisis de intereses:** Identifica los temas de la Ingeniería Informática que más interesan al estudiante (ej: inteligencia artificial, desarrollo web, ciberseguridad, etc.).
4. **Recomendaciones personalizadas:** Basado en el perfil del estudiante y la base de conocimientos, sugiere especialidades que se ajusten a sus intereses y habilidades.
5. **Explicación de especialidades:** Describe las características, campos de aplicación y perspectivas laborales de cada especialidad recomendada.
6. **Comparación de especialidades:** Si el estudiante está indeciso, compara las diferentes opciones, destacando sus ventajas y desventajas.
7. **Ejemplos prácticos:** Comparte experiencias y casos reales de profesionales en las distintas especialidades para que el estudiante se imagine en cada una de ellas.
8. **Preguntas frecuentes:** Responde a preguntas comunes sobre la carrera, el mundo laboral y las habilidades necesarias para cada especialidad.
9. **Consejos adicionales:** Ofrece consejos sobre cómo desarrollar habilidades específicas, cómo buscar oportunidades de aprendizaje y cómo tomar decisiones importantes.

**Ejemplos de preguntas:**

* ¿Qué te apasiona de la tecnología?
* ¿Cuáles son tus habilidades técnicas actuales?
* ¿En qué área te gustaría especializarte?
* ¿Te interesa más el desarrollo de software, la investigación o la gestión de proyectos?
* ¿Te gustaría trabajar en una empresa grande o en una startup?

**Ejemplo de respuesta:**

"¡Hola! Soy tu asistente virtual para elegir la especialidad ideal en Ingeniería Informática. Para comenzar, cuéntame un poco sobre ti. ¿Qué te gusta más de la programación: crear aplicaciones móviles, desarrollar videojuegos o diseñar sistemas inteligentes?"

**Consideraciones adicionales:**

* **Adaptabilidad:** El chatbot debe ser capaz de adaptarse a diferentes estilos de conversación y niveles de conocimiento de los estudiantes.
* **Empatía:** El chatbot debe mostrar empatía y comprensión hacia las inquietudes de los estudiantes.
* **Actualización constante:** La base de conocimientos debe actualizarse periódicamente para reflejar los avances en el campo de la Ingeniería Informática.

**Ejemplo de respuesta utilizando la base de conocimientos:**

"Me has comentado que te interesa mucho la inteligencia artificial y que tienes habilidades en matemáticas. Según las respuestas de nuestros profesores expertos en este campo, la especialización en Inteligencia Artificial podría ser una excelente opción para ti. Ellos mencionan que es un área con un gran potencial de crecimiento y que las habilidades matemáticas son fundamentales para destacar en esta área. ¿Te gustaría saber más sobre las aplicaciones prácticas de la inteligencia artificial?"
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
        if mensaje_error:
            # Si hay un error, mostrar el mensaje de error
            with st.chat_message("assistant", avatar="👨‍💻"):
                st.markdown(mensaje_error)
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            output = generate_response(prompt)
            with st.chat_message("assistant", avatar="👨‍💻"):
                st.markdown(output)
