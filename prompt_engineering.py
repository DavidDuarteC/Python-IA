"""
PROMPT ENGINEERING - Técnicas clave para entrevista Redarbor
============================================================
Las 5 técnicas más importantes que debes conocer:
1. Zero-shot
2. Few-shot
3. Chain-of-Thought (CoT)
4. System prompt con rol
5. Output estructurado (JSON)
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llamar_llm(sistema: str, usuario: str, temperatura: float = 0.1) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sistema},
            {"role": "user", "content": usuario}
        ],
        temperature=temperatura
    )
    return response.choices[0].message.content


# ── TÉCNICA 1: Zero-shot ───────────────────────────────────
def ejemplo_zero_shot():
    """Sin ejemplos — el modelo usa solo su conocimiento base"""
    print("\n1️⃣  ZERO-SHOT")
    sistema = "Eres un evaluador de CVs para Computrabajo."
    usuario = "¿Este candidato hace match con una oferta de Python Developer? CV: 3 años en Django, PostgreSQL, REST APIs."
    
    respuesta = llamar_llm(sistema, usuario)
    print(f"Respuesta: {respuesta[:200]}...")


# ── TÉCNICA 2: Few-shot ────────────────────────────────────
def ejemplo_few_shot():
    """Con ejemplos — el modelo aprende el formato esperado"""
    print("\n2️⃣  FEW-SHOT")
    sistema = """Eres un clasificador de urgencia de ofertas de empleo.
    
Ejemplos:
CV: "Buscamos cirujano cardíaco para UCI" → Urgencia: ALTA
CV: "Se necesita community manager para redes sociales" → Urgencia: MEDIA  
CV: "Buscamos pasante de recursos humanos" → Urgencia: BAJA

Clasifica la siguiente oferta siguiendo exactamente el mismo formato."""
    
    usuario = "Se requiere ingeniero de sistemas para soporte crítico de producción 24/7"
    
    respuesta = llamar_llm(sistema, usuario)
    print(f"Respuesta: {respuesta}")


# ── TÉCNICA 3: Chain-of-Thought ────────────────────────────
def ejemplo_chain_of_thought():
    """Le pides al modelo que razone paso a paso antes de responder"""
    print("\n3️⃣  CHAIN-OF-THOUGHT")
    sistema = """Eres un headhunter experto. Cuando evalúes candidatos, 
    SIEMPRE sigue este proceso paso a paso:
    1. Analiza los requisitos de la oferta
    2. Identifica las habilidades del candidato
    3. Compara requisitos vs habilidades
    4. Calcula el porcentaje de match
    5. Da tu recomendación final
    
    Muestra TODOS los pasos en tu respuesta."""
    
    usuario = """
    Oferta: Senior Python Developer - Requiere: Python 5+ años, AWS, Docker, Kubernetes, CI/CD
    Candidato: Python 6 años, AWS certified, Docker (2 años), sin experiencia en Kubernetes
    """
    
    respuesta = llamar_llm(sistema, usuario, temperatura=0.2)
    print(f"Respuesta:\n{respuesta[:400]}...")


# ── TÉCNICA 4: Output estructurado JSON ────────────────────
def ejemplo_output_json():
    """Forzar al modelo a responder en formato JSON específico"""
    print("\n4️⃣  OUTPUT ESTRUCTURADO (JSON)")
    sistema = """Eres un evaluador de CVs. Responde ÚNICAMENTE en formato JSON válido,
    sin texto adicional antes ni después. El formato debe ser exactamente:
    {
        "match_porcentaje": número del 0 al 100,
        "habilidades_match": ["lista", "de", "habilidades", "que", "coinciden"],
        "habilidades_faltantes": ["lista", "de", "lo", "que", "falta"],
        "recomendacion": "CONTRATAR" | "ENTREVISTAR" | "RECHAZAR",
        "razon": "explicación breve en máximo 50 palabras"
    }"""
    
    usuario = """
    Oferta: AI Engineer - Requiere: Python, LangChain, RAG, embeddings, OpenAI API
    CV: Python 3 años, OpenAI API, n8n, MCP, agentes AI, PostgreSQL, GCP
    """
    
    respuesta = llamar_llm(sistema, usuario, temperatura=0.0)
    
    try:
        datos = json.loads(respuesta)
        print(f"JSON parseado correctamente:")
        print(json.dumps(datos, indent=2, ensure_ascii=False))
    except:
        print(f"Respuesta raw: {respuesta}")


# ── TÉCNICA 5: Reducir alucinaciones ──────────────────────
def ejemplo_anti_alucinacion():
    """Técnicas para que el modelo no invente información"""
    print("\n5️⃣  ANTI-ALUCINACIÓN")
    sistema = """Eres un asistente de Redarbor. Reglas ESTRICTAS:
    - Responde SOLO con información del contexto proporcionado
    - Si la información no está en el contexto, di exactamente: "No tengo esa información en mi base de datos."
    - NUNCA inventes datos, estadísticas o nombres
    - NUNCA asumas información que no esté explícitamente en el contexto"""
    
    contexto = """
    CONTEXTO:
    - Redarbor tiene 1.100 colaboradores
    - Fundada en 2013
    - Tiene oficinas en 13 países
    - Computrabajo es líder en LATAM
    """
    
    # Pregunta que SÍ está en el contexto
    pregunta1 = f"{contexto}\n\nPregunta: ¿Cuántos colaboradores tiene Redarbor?"
    r1 = llamar_llm(sistema, pregunta1)
    print(f"Pregunta en contexto: {r1}")
    
    # Pregunta que NO está en el contexto
    pregunta2 = f"{contexto}\n\nPregunta: ¿Cuál es el revenue anual de Redarbor?"
    r2 = llamar_llm(sistema, pregunta2)
    print(f"Pregunta fuera de contexto: {r2}")


# ── MAIN ───────────────────────────────────────────────────
def main():
    print("PROMPT ENGINEERING - 5 técnicas clave")
    print("=" * 50)
    print("NOTA: Necesitas OPENAI_API_KEY configurada para ejecutar")
    print("=" * 50)
    
    ejemplo_zero_shot()
    ejemplo_few_shot()
    ejemplo_chain_of_thought()
    ejemplo_output_json()
    ejemplo_anti_alucinacion()


if __name__ == "__main__":
    main()