import streamlit as st
import pymongo
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Chatbot de Películas",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    }
    .stTitle {
        color: #f8fafc;
    }
    .stSidebar {
        background-color: #0f172a;
    }
    .stButton>button {
        background-color: #f97316;
        color: white;
        border: none;
    }
    .stTextInput>div>div>input {
        background: #1f2937;
        color: #f8fafc;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3 {
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================
# CONFIGURACIÓN
# =======================
GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

# =======================
# CLIENTES (cacheados)
# =======================
@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["movies_db"]
    return db["movies_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# UTILIDADES
# =======================
def crear_embedding(texto: str):
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values


def buscar_similares(embedding, k=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:
    contexto = "\n\n".join([f"- {c['texto']}" for c in contextos])
    prompt = f"""Eres un asistente experto en películas.
Usa exclusivamente el contexto provisto para responder la pregunta del usuario.
Si la respuesta no está en el contexto, dilo claramente.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde en español, de forma clara y breve."""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# =======================
# INTERFAZ STREAMLIT
# =======================

st.title("🎬 Chatbot de Películas")
st.markdown("## Responde preguntas sobre directores, actores, géneros y sinopsis.")
st.write("Escribe tu consulta y el asistente buscará en el corpus de películas cargado en MongoDB.")
st.divider()

with st.sidebar:
    st.header("Bienvenido")
    st.write("Este chat usa un corpus de películas en MongoDB para responder mejor.")
    st.markdown("**Colección esperada:** `movies_db.movies_vectors`")
    st.markdown("**Índice esperado:** `vector_index` sobre `embedding`")
    st.markdown("---")
    st.subheader("Ejemplos rápidos")
    st.write("- ¿Quién dirigió *Pulp Fiction*?")
    st.write("- ¿Qué películas protagoniza Scarlett Johansson?")
    st.write("- Describe la trama de *Inception*.")
    st.write("- Películas similares a *The Matrix*.")
    st.markdown("---")
    if st.button("Reiniciar conversación"):
        st.session_state.historial = []
        st.experimental_rerun()

if "historial" not in st.session_state:
    st.session_state.historial = []

col1, col2, col3 = st.columns([2, 1, 1])
col1.metric("Corpus", "Películas")
col2.metric("Modelo", "Gemini 2.5")
col3.metric("Mensajes", len(st.session_state.historial))

if not st.session_state.historial:
    st.info("Escribe una pregunta en la caja de chat para iniciar.")

for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])

pregunta = st.chat_input("¿Qué quieres saber sobre películas?")

if pregunta:
    st.session_state.historial.append({"rol": "usuario", "texto": pregunta})
    st.chat_message("user").write(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el corpus de películas..."):
            try:
                emb = crear_embedding(pregunta)
                similares = buscar_similares(emb, k=5)
                if not similares:
                    respuesta = "No encontré información relevante sobre películas en el corpus."
                else:
                    respuesta = generar_respuesta(pregunta, similares)
            except Exception as e:
                respuesta = f"⚠️ Ocurrió un error: {e}"

        st.write(respuesta)

        if 'similares' in locals() and similares:
            with st.expander("🔍 Fragmentos usados para generar la respuesta"):
                for i, c in enumerate(similares, 1):
                    st.markdown(f"**Fragmento {i}** — score: `{c['score']:.4f}`")
                    st.write(c["texto"][:500] + ("…" if len(c["texto"]) > 500 else ""))
                    st.divider()

    st.session_state.historial.append({"rol": "bot", "texto": respuesta})
