"""
RAG BÁSICO - Práctica para entrevista técnica Redarbor.
======================================================
Flujo:
1. Cargar documentos
2. Hacer chunking
3. Generar embeddings con OpenAI
4. Guardar en ChromaDB (base de datos vectorial)
5. En cada query: buscar chunks relevantes
6. Inyectar al prompt y llamar al LLM
"""

import json  # librería estándar para parsear y serializar JSON
import math  # librería estándar para operaciones matemáticas como sqrt
import os  # librería estándar para la interacción con el sistema operativo y variables de entorno
import re  # librería estándar para expresiones regulares y tokenización de cadenas
import zlib  # librería estándar para hashing determinista de tokens

try:
    import requests  # cliente HTTP de terceros más cómodo si está instalado
except ImportError:
    requests = None  # si no está instalado, usaremos urllib en su lugar

try:
    import chromadb  # motor vectorial local para embeddings y búsqueda semántica
except ImportError:
    chromadb = None  # si no está instalado, la aplicación usará un fallback local

# ── Configuración ──────────────────────────────────────────
# Leer variables de entorno desde un archivo .env opcional.
# Este script asume que existe .env con la variable OPENCODE_API_KEY.

# Cargar variables de entorno desde .env sin depender de librerías externas.
def cargar_env(path=".env"):
    if not os.path.exists(path):  # si el archivo .env no existe, no hacemos nada
        return
    with open(path, "r", encoding="utf-8") as f:  # abrir .env en modo lectura
        for linea in f:  # recorrer cada línea del archivo
            linea = linea.strip()  # quitar espacios en blanco al inicio y final
            if not linea or linea.startswith("#"):  # ignorar líneas vacías o comentarios
                continue
            if "=" not in linea:  # ignorar líneas que no contienen asignaciones válidas
                continue
            clave, valor = linea.split("=", 1)  # separar nombre y valor por el primer '='
            clave = clave.strip()  # limpiar espacios en la clave
            valor = valor.strip().strip('"').strip("'")  # limpiar espacios y comillas del valor
            if clave and valor and clave not in os.environ:  # solo asignar si no existe ya en el entorno
                os.environ[clave] = valor  # establecer la variable de entorno

cargar_env()  # cargar variables de entorno desde .env antes de leerlas

OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")  # leer la clave API de Opencode del entorno
OPENCODE_API_BASE = os.getenv("OPENCODE_API_BASE", "https://opencode.ai/zen/go/v1")  # URL base del API
OPENCODE_CHAT_MODEL = os.getenv("OPENCODE_CHAT_MODEL", "minimax-m2.7")  # modelo de chat a usar

if not OPENCODE_API_KEY:  # validar que exista la clave API
    raise ValueError(
        "Falta OPENCODE_API_KEY. Define la variable de entorno o agrega .env con la clave."
    )  # detener ejecución si no hay clave


def opencode_post(path, payload):
    """Enviar una petición POST a la API de Opencode con el payload dado."""
    url = OPENCODE_API_BASE.rstrip("/") + path  # construir la URL sin barras duplicadas
    headers = {
        "Content-Type": "application/json",  # indicar JSON en el body
        "Authorization": f"Bearer {OPENCODE_API_KEY}",  # token en cabecera Authorization
        "X-API-Key": OPENCODE_API_KEY,  # cabecera alternativa para compatibilidad
        "User-Agent": "Python-Opencode-Client/1.0",  # identificador del cliente
    }
    if requests:  # si requests está instalado, usarlo para la petición
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:  # si la respuesta no es 200 OK
            raise RuntimeError(
                f"Error en Opencode API {response.status_code}: {response.reason}\n"
                f"URL: {url}\n"
                f"Revisa tu OPENCODE_API_KEY, OPENCODE_API_BASE y que tu clave tenga acceso al modelo.\n"
                f"Respuesta del servidor:\n{response.text}"
            )  # lanzar error detallado para debugging
        return response.json()  # devolver el JSON parseado de la respuesta

    # Si requests no está disponible, usar urllib estándar de Python.
    data = json.dumps(payload).encode("utf-8")  # serializar payload a bytes
    import urllib.error  # módulo estándar para errores HTTP
    import urllib.request  # módulo estándar para peticiones HTTP

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )  # construir la petición HTTP manualmente

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))  # parsear JSON de la respuesta
    except urllib.error.HTTPError as exc:  # capturar errores HTTP
        body = exc.read().decode("utf-8", errors="ignore")  # leer el contenido del error
        mensaje = (
            f"Error en Opencode API {exc.code}: {exc.reason}\n"
            f"URL: {url}\n"
            f"Revisa tu OPENCODE_API_KEY, OPENCODE_API_BASE y que tu clave tenga acceso al modelo.\n"
            f"Respuesta del servidor:\n{body}"
        )  # construir mensaje de error completo
        raise RuntimeError(mensaje)  # lanzar el error con detalles


# ── 1. DOCUMENTOS DE EJEMPLO ───────────────────────────────
# En un caso real, estos documentos vendrían de archivos PDF, bases de datos o APIs.
documentos = [
    "Redarbor es la compañía HR Tech número 1 en Latinoamérica. Fue fundada en 2013 y tiene más de 1.100 colaboradores en 13 oficinas.",
    "Computrabajo es el portal de empleo líder en Latinoamérica y forma parte del grupo Redarbor. Ayuda a más de 50 millones de personas a encontrar empleo cada mes.",
    "Pandapé es el software SaaS de RRHH de Redarbor. Tiene más de 8.000 clientes activos y es líder en la región.",
    "InfoJobs es el portal de empleo número 1 en Brasil y también pertenece al grupo Redarbor.",
    "Redarbor ha sido reconocida como Great Place to Work en España y recibió mención especial Better for Business.",
    "OCC es el portal de empleo número 1 en México y forma parte del ecosistema de Redarbor.",
]


# ── 2. CHUNKING ────────────────────────────────────────────
# Esta función corta textos largos en fragmentos manejables para la búsqueda.
def hacer_chunks(textos, max_chars=500):
    """Divide textos largos en trozos más pequeños usando frases como límite."""
    chunks = []  # lista donde guardamos los fragmentos
    for texto in textos:  # recorrer cada texto de entrada
        if len(texto) <= max_chars:  # si ya es corto, no hace falta dividir
            chunks.append(texto)  # guardar el texto tal cual
        else:
            palabras = texto.split(". ")  # dividir el texto en frases por punto seguido
            chunk_actual = ""  # acumulador del chunk en construcción
            for palabra in palabras:  # procesar cada frase
                if len(chunk_actual) + len(palabra) <= max_chars:
                    chunk_actual += palabra + ". "  # añadir la frase al chunk actual
                else:
                    chunks.append(chunk_actual.strip())  # cerrar y guardar el chunk actual
                    chunk_actual = palabra + ". "  # iniciar nuevo chunk con la frase actual
            if chunk_actual:  # si queda texto pendiente al final
                chunks.append(chunk_actual.strip())  # agregar el último chunk
    return chunks  # devolver la lista completa de chunks


# ── 3. EMBEDDINGS SIMPLES ───────────────────────────────────
# Generar un embedding local simple para poder buscar sin un servicio externo.
def texto_a_vector(texto, dim=128):
    """Convierte el texto en un vector numérico simple usando hashing de tokens."""
    vector = [0.0] * dim  # inicializar un vector de dimensión fija con ceros
    tokens = re.findall(r"\w+", texto.lower())  # tokenizar el texto en minúsculas
    for token in tokens:  # cada token contribuye al vector
        idx = zlib.adler32(token.encode("utf-8")) % dim  # obtener un índice determinista
        vector[idx] += 1.0  # contar la frecuencia del token en ese índice

    norma = math.sqrt(sum(v * v for v in vector))  # calcular la longitud del vector
    if norma == 0:  # si no hay tokens guardados
        return vector  # devolver el vector sin normalizar
    return [v / norma for v in vector]  # normalizar para comparaciones de similitud


def buscar_chunks_relevantes_local(chunks, query, n_resultados=3):
    """Buscar los chunks más relevantes usando coincidencias de tokens simples."""
    query_tokens = set(re.findall(r"\w+", query.lower()))  # tokens de la consulta
    scored = []  # lista temporal de (puntuación, chunk)
    for chunk in chunks:  # evaluar cada chunk del documento
        chunk_tokens = set(re.findall(r"\w+", chunk.lower()))  # tokens del chunk
        score = len(query_tokens & chunk_tokens)  # contar tokens comunes con la consulta
        scored.append((score, chunk))  # guardar la puntuación y el fragmento
    scored.sort(key=lambda item: item[0], reverse=True)  # ordenar de mayor a menor pertinencia
    relevantes = [chunk for score, chunk in scored if score > 0]  # quedarse con chunks coincidentes
    return relevantes[:n_resultados] if relevantes else chunks[:n_resultados]  # devolver top N o fallback


# ── 4. ÍNDICE VECTORIAL DE CHROMA ───────────────────────────
# Crear un índice vectorial usando ChromaDB si está instalado.
def crear_base_vectorial(chunks):
    """Crea una colección de ChromaDB con embeddings de los chunks."""
    print("Construyendo índice local de chunks con ChromaDB...")  # mensaje de inicio
    if chromadb is None:  # si ChromaDB no está disponible
        print("WARNING: chromadb no está instalado. Usando búsqueda local en su lugar.")
        return chunks  # usar los chunks sin indexar como fallback

    client = chromadb.Client()  # crear cliente de ChromaDB
    nombre = "redarbor_rag"  # nombre de la colección local

    try:
        existing = [c.name for c in client.list_collections()]  # listar colecciones existentes
        if nombre in existing:
            client.delete_collection(name=nombre)  # eliminar colección anterior si existe
    except Exception:
        pass  # ignorar errores de listado/eliminación

    collection = client.create_collection(name=nombre)  # crear la colección nueva
    ids = [f"chunk_{i}" for i in range(len(chunks))]  # identificar cada chunk de forma única
    embeddings = [texto_a_vector(chunk) for chunk in chunks]  # generar embeddings locales para cada chunk
    metadatas = [{"source": "doc", "index": i} for i in range(len(chunks))]  # metadatos opcionales por chunk

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )  # agregar los chunks a la colección vectorial
    return collection  # devolver el objeto collection creado


def buscar_chunks_relevantes(coleccion, query, n_resultados=3):
    """Recuperar los chunks relevantes usando ChromaDB o el fallback local."""
    if chromadb is None or not hasattr(coleccion, "query"):
        return buscar_chunks_relevantes_local(coleccion, query, n_resultados)  # fallback si no hay ChromaDB

    query_embedding = texto_a_vector(query)  # embedding de la pregunta
    resultado = coleccion.query(
        query_embeddings=[query_embedding],
        n_results=n_resultados,
        include=["documents"],
    )  # ejecutar consulta vectorial
    documentos = resultado.get("documents", [[]])[0]  # extraer la lista de documentos devueltos
    return documentos if documentos else []  # devolver documentos relevantes o vacío


# ── 5. GENERACIÓN CON CONTEXTO ─────────────────────────────
# Combina la recuperación de contexto con la generación del modelo.
def responder_con_rag(coleccion, pregunta):
    """Generar una respuesta del modelo usando chunks relevantes como contexto."""
    chunks_relevantes = buscar_chunks_relevantes(coleccion, pregunta)  # recuperar contexto relevante
    contexto = "\n".join(chunks_relevantes)  # unir los chunks en un solo bloque de texto

    prompt_sistema = """Eres un asistente útil. Responde ÚNICAMENTE basándote en el contexto 
proporcionado. Si la información no está en el contexto, di explícitamente 
'No tengo información suficiente para responder eso.'
No inventes datos."""  # instrucciones del sistema para el modelo

    prompt_usuario = f"""Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""  # prompt del usuario que incluye contexto y pregunta

    payload = {
        "model": OPENCODE_CHAT_MODEL,  # modelo para la llamada de chat
        "messages": [
            {"role": "system", "content": prompt_sistema},  # mensaje de sistema
            {"role": "user", "content": prompt_usuario},  # mensaje del usuario con el prompt completo
        ],
        "temperature": 0.1,  # baja temperatura para respuestas más consistentes
    }
    result = opencode_post("/messages", payload)  # llamar al endpoint de chat de Opencode

    def extraer_contenido(res):
        """Extraer el texto principal de la estructura de respuesta devuelta por la API."""
        if not isinstance(res, dict):
            return ""  # si la respuesta no es un diccionario, no hay texto válido

        if "choices" in res and isinstance(res["choices"], list):  # formato similar a OpenAI
            for choice in res["choices"]:
                if isinstance(choice, dict):
                    msg = choice.get("message")  # extraer posible objeto message
                    if isinstance(msg, dict) and msg.get("content"):
                        return msg["content"]  # devolver contenido de message
                    out = choice.get("output")  # extraer posible objeto output
                    if isinstance(out, dict) and out.get("content"):
                        return out["content"]  # devolver contenido de output
                    if isinstance(out, list):
                        texts = [item.get("content") for item in out if isinstance(item, dict) and item.get("content")]
                        if texts:
                            return "\n".join(texts)  # unir múltiples contenidos

        if "output" in res:  # formato alternativo con output en la raíz
            out = res["output"]
            if isinstance(out, dict) and out.get("content"):
                return out["content"]  # devolver contenido de output
            if isinstance(out, list):
                texts = []
                for item in out:
                    if isinstance(item, dict) and item.get("content"):
                        texts.append(item["content"])
                if texts:
                    return "\n".join(texts)  # unir múltiples contenidos

        msg = res.get("message")  # mensaje directo en la raíz
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]  # devolver contenido de message raíz

        if "result" in res and isinstance(res["result"], dict) and res["result"].get("output"):
            return res["result"]["output"].get("content", "")  # caso con estructura result -> output

        return ""  # no se encontró texto válido

    contenido = extraer_contenido(result)  # extraer el texto principal de la respuesta
    if not contenido:  # si no se pudo extraer contenido
        print("[DEBUG] Respuesta de Opencode sin texto. Resultado completo:")
        print(json.dumps(result, indent=2, ensure_ascii=False))  # imprimir la respuesta completa para debugging
        contenido = "No pude obtener contenido del modelo. Revisa el resultado en la salida de debug."  # fallback de texto

    return {
        "respuesta": contenido,  # respuesta lista para mostrar al usuario
        "contexto_usado": chunks_relevantes,  # chunks que se utilizaron para generar la respuesta
    }


# ── MAIN: todo junto ───────────────────────────────────────
# Ejecutar el flujo completo de la demo RAG.
def main():
    print("=" * 50)  # separador visual de inicio
    print("RAG BÁSICO - Demo")  # título principal impreso
    print("=" * 50)  # separador visual

    chunks = hacer_chunks(documentos)  # generar chunks a partir de los documentos
    print(f"\n✓ {len(chunks)} chunks generados a partir de {len(documentos)} documentos")  # informar cantidad de chunks

    coleccion = crear_base_vectorial(chunks)  # crear índice vectorial con los chunks
    print(f"\n✓ Base vectorial lista con {len(chunks)} chunks\n")  # confirmar la creación de la base

    preguntas = [
        "¿Cuántos colaboradores tiene Redarbor?",
        "¿Qué es Pandapé?",
        "¿Cuál es el portal de empleo número 1 en México?",
        "¿Cuánto cuesta la suscripción a Computrabajo?",  # pregunta no cubierta en los documentos de ejemplo
    ]

    print("=" * 50)  # separador para la sección de preguntas
    print("PREGUNTAS AL SISTEMA RAG")  # encabezado de la sección de preguntas
    print("=" * 50)  # separador para la sección de preguntas

    for pregunta in preguntas:  # iterar cada pregunta de muestra
        print(f"\n❓ {pregunta}")  # imprimir la pregunta actual
        resultado = responder_con_rag(coleccion, pregunta)  # generar respuesta usando el pipeline RAG
        print(f"💬 {resultado['respuesta']}")  # imprimir la respuesta generada
        print(f"📄 Contexto usado: {resultado['contexto_usado'][0][:80]}...")  # mostrar el primer chunk usado como contexto


if __name__ == "__main__":
    main()  # ejecutar la función main al correr el script directamente
