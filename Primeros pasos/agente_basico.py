"""
AGENTE AI BÁSICO CON TOOLS - Práctica para entrevista técnica Redarbor
======================================================================
Un agente es un LLM que puede:
1. Recibir una pregunta
2. Decidir qué herramienta usar
3. Ejecutar la herramienta
4. Usar el resultado para responder

Flujo: Query → LLM decide tool → ejecuta tool → LLM genera respuesta final
"""

import os
import json

try:
    import requests
except ImportError:
    requests = None

def cargar_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            if "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            clave = clave.strip()
            valor = valor.strip().strip('"').strip("'")
            if clave and valor and clave not in os.environ:
                os.environ[clave] = valor

cargar_env()

OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_API_BASE = os.getenv("OPENCODE_API_BASE", "https://opencode.ai/zen/go/v1")
OPENCODE_CHAT_MODEL = os.getenv("OPENCODE_CHAT_MODEL", "minimax-m2.7")

if not OPENCODE_API_KEY:
    raise ValueError(
        "Falta OPENCODE_API_KEY. Define la variable de entorno o agrega .env con la clave."
    )


def opencode_post(path, payload):
    """Enviar una petición POST a la API de Opencode con el payload dado."""
    url = OPENCODE_API_BASE.rstrip("/") + path
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENCODE_API_KEY}",
        "X-API-Key": OPENCODE_API_KEY,
        "User-Agent": "Python-Opencode-Client/1.0",
    }
    if requests:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Error en Opencode API {response.status_code}: {response.reason}\n"
                f"URL: {url}\n"
                f"Respuesta del servidor:\n{response.text}"
            )
        return response.json()

    data = json.dumps(payload).encode("utf-8")
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        mensaje = (
            f"Error en Opencode API {exc.code}: {exc.reason}\n"
            f"URL: {url}\n"
            f"Respuesta del servidor:\n{body}"
        )
        raise RuntimeError(mensaje)


def extraer_contenido(res):
    """Extraer el texto principal de la estructura de respuesta devuelta por la API."""
    if not isinstance(res, dict):
        return ""
    if "choices" in res and isinstance(res["choices"], list):
        for choice in res["choices"]:
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict) and msg.get("content"):
                    return msg["content"]
                out = choice.get("output")
                if isinstance(out, dict) and out.get("content"):
                    return out["content"]
                if isinstance(out, list):
                    texts = [item.get("content") for item in out if isinstance(item, dict) and item.get("content")]
                    if texts:
                        return "\n".join(texts)
    if "output" in res:
        out = res["output"]
        if isinstance(out, dict) and out.get("content"):
            return out["content"]
        if isinstance(out, list):
            texts = []
            for item in out:
                if isinstance(item, dict) and item.get("content"):
                    texts.append(item["content"])
            if texts:
                return "\n".join(texts)
    msg = res.get("message")
    if isinstance(msg, dict) and msg.get("content"):
        return msg["content"]
    if "result" in res and isinstance(res["result"], dict) and res["result"].get("output"):
        return res["result"]["output"].get("content", "")
    return ""


# ── TOOLS (herramientas que el agente puede usar) ──────────
def buscar_empleo(titulo: str, ciudad: str = "Bogotá") -> dict:
    """Simula búsqueda de empleos en Computrabajo"""
    empleos_mock = [
        {"titulo": f"{titulo} Senior", "empresa": "TechCorp", "ciudad": ciudad, "salario": "8-12M COP"},
        {"titulo": f"{titulo} Junior", "empresa": "StartupXYZ", "ciudad": ciudad, "salario": "3-5M COP"},
        {"titulo": f"{titulo} Lead", "empresa": "BigCo", "ciudad": ciudad, "salario": "15-20M COP"},
    ]
    return {"resultados": empleos_mock, "total": len(empleos_mock)}


def obtener_salario_promedio(cargo: str, ciudad: str = "Colombia") -> dict:
    """Simula obtención de salario promedio por cargo"""
    salarios_mock = {
        "desarrollador": {"min": 3000000, "max": 12000000, "promedio": 6500000},
        "data scientist": {"min": 5000000, "max": 18000000, "promedio": 10000000},
        "devops": {"min": 4000000, "max": 15000000, "promedio": 8000000},
        "ai engineer": {"min": 6000000, "max": 26000000, "promedio": 12000000},
    }
    cargo_lower = cargo.lower()
    for key in salarios_mock:
        if key in cargo_lower:
            return {"cargo": cargo, "ciudad": ciudad, **salarios_mock[key]}
    return {"cargo": cargo, "ciudad": ciudad, "min": 3000000, "max": 10000000, "promedio": 5000000}


def contar_ofertas_activas(categoria: str) -> dict:
    """Simula conteo de ofertas activas en Computrabajo"""
    conteos_mock = {
        "tecnologia": 15420,
        "marketing": 8230,
        "ventas": 22100,
        "rrhh": 5600,
        "finanzas": 9800,
    }
    cat_lower = categoria.lower()
    count = conteos_mock.get(cat_lower, 3000)
    return {"categoria": categoria, "ofertas_activas": count}


FUNCIONES_DISPONIBLES = {
    "buscar_empleo": buscar_empleo,
    "obtener_salario_promedio": obtener_salario_promedio,
    "contar_ofertas_activas": contar_ofertas_activas,
}


TOOLS_DESCRIPTION = """
Tienes acceso a las siguientes herramientas:
1. buscar_empleo(titulo, ciudad) - Busca ofertas de empleo por título y ciudad en Computrabajo
2. obtener_salario_promedio(cargo, ciudad) - Obtiene el rango de salario promedio para un cargo en Colombia
3. contar_ofertas_activas(categoria) - Cuenta cuántas ofertas activas hay en una categoría en Computrabajo

Responde SIEMPRE usando JSON con el formato:
{
    "necesita_tool": true/false,
    "tool": "nombre_de_la_tool" (si necesita_tool es true),
    "argumentos": {"param1": "valor1"} (si necesita_tool es true),
    "respuesta": "respuesta directa" (si necesita_tool es false)
}
"""


def ejecutar_agente(pregunta: str, verbose: bool = True) -> str:
    """
    Agente que usa tools para responder preguntas.
    
    El LLM decide qué tool usar, nosotros la ejecutamos,
    y el LLM usa el resultado para dar la respuesta final.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"❓ Pregunta: {pregunta}")
        print(f"{'='*50}")
    
    max_iteraciones = 5
    mensajes = [
        {
            "role": "system",
            "content": f"""Eres un asistente de Redarbor/Computrabajo especializado en mercado laboral colombiano.
            Ayudas a personas a encontrar empleo y entender el mercado de trabajo.
            {TOOLS_DESCRIPTION}
            IMPORTANTE: Responde siempre con JSON válido según el formato indicado arriba."""
        },
        {"role": "user", "content": pregunta}
    ]
    
    for iteracion in range(max_iteraciones):
        payload = {
            "model": OPENCODE_CHAT_MODEL,
            "messages": mensajes,
            "temperature": 0.1,
        }
        result = opencode_post("/messages", payload)
        contenido = extraer_contenido(result)
        
        if not contenido:
            if verbose:
                print("[DEBUG] Respuesta vacía de Opencode")
            return "No pude obtener respuesta del modelo."
        
        if verbose:
            print(f"\n🔍 Decisión del agente: {contenido[:200]}...")
        
        try:
            decision = json.loads(contenido)
        except json.JSONDecodeError:
            if verbose:
                print("⚠️ No se pudo parsear JSON, interpretando como respuesta directa")
            if verbose:
                print(f"\n💬 Respuesta: {contenido}")
            return contenido
        
        if decision.get("necesita_tool") and decision.get("tool"):
            nombre_tool = decision.get("tool")
            argumentos = decision.get("argumentos", {})
            
            if verbose:
                print(f"\n🔧 Usando tool: {nombre_tool}")
                print(f"   Argumentos: {argumentos}")
            
            if nombre_tool in FUNCIONES_DISPONIBLES:
                resultado = FUNCIONES_DISPONIBLES[nombre_tool](**argumentos)
            else:
                resultado = {"error": f"Tool {nombre_tool} no encontrada"}
            
            if verbose:
                print(f"   Resultado: {json.dumps(resultado, ensure_ascii=False)[:150]}...")
            
            mensajes.append({"role": "assistant", "content": contenido})
            mensajes.append({
                "role": "user",
                "content": f"El resultado de la tool fue: {json.dumps(resultado)}. Ahora genera la respuesta final para el usuario."
            })
        else:
            respuesta = decision.get("respuesta", contenido)
            if verbose:
                print(f"\n💬 Respuesta: {respuesta}")
            return respuesta
    
    return "No pude completar la tarea en el número máximo de iteraciones."


# ── MAIN ───────────────────────────────────────────────────
def main():
    print("AGENTE AI - Demo Redarbor")
    print("=" * 50)
    
    preguntas = [
        "¿Cuánto gana un AI Engineer en Colombia?",
        "Busca empleos de Data Scientist en Medellín",
        "¿Cuántas ofertas de tecnología hay activas en Computrabajo?",
        "Quiero saber el salario de un desarrollador y cuántas ofertas hay en tecnología",
    ]
    
    for pregunta in preguntas:
        ejecutar_agente(pregunta, verbose=True)
        print()


if __name__ == "__main__":
    main()