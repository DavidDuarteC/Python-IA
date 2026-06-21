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
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── TOOLS (herramientas que el agente puede usar) ──────────
def buscar_empleo(titulo: str, ciudad: str = "Bogotá") -> dict:
    """Simula búsqueda de empleos en Computrabajo"""
    # En producción esto llamaría a una API real
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


# ── DEFINICIÓN DE TOOLS PARA OPENAI ───────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_empleo",
            "description": "Busca ofertas de empleo por título y ciudad en Computrabajo",
            "parameters": {
                "type": "object",
                "properties": {
                    "titulo": {"type": "string", "description": "Título del cargo a buscar"},
                    "ciudad": {"type": "string", "description": "Ciudad donde buscar", "default": "Bogotá"}
                },
                "required": ["titulo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_salario_promedio",
            "description": "Obtiene el rango de salario promedio para un cargo en Colombia",
            "parameters": {
                "type": "object",
                "properties": {
                    "cargo": {"type": "string", "description": "Nombre del cargo"},
                    "ciudad": {"type": "string", "description": "Ciudad o país", "default": "Colombia"}
                },
                "required": ["cargo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "contar_ofertas_activas",
            "description": "Cuenta cuántas ofertas activas hay en una categoría en Computrabajo",
            "parameters": {
                "type": "object",
                "properties": {
                    "categoria": {"type": "string", "description": "Categoría de empleo (tecnologia, marketing, ventas, etc.)"}
                },
                "required": ["categoria"]
            }
        }
    }
]

# Mapa de funciones disponibles
FUNCIONES_DISPONIBLES = {
    "buscar_empleo": buscar_empleo,
    "obtener_salario_promedio": obtener_salario_promedio,
    "contar_ofertas_activas": contar_ofertas_activas,
}


# ── EL AGENTE ──────────────────────────────────────────────
def ejecutar_agente(pregunta: str, verbose: bool = True) -> str:
    """
    Agente que usa tools para responder preguntas.
    
    El LLM decide qué tool usar, nosotros la ejecutamos,
    y el LLM usa el resultado para dar la respuesta final.
    """
    mensajes = [
        {
            "role": "system",
            "content": """Eres un asistente de Redarbor/Computrabajo especializado en mercado laboral colombiano.
            Ayudas a personas a encontrar empleo y entender el mercado de trabajo.
            Usa las herramientas disponibles para dar respuestas precisas y actualizadas."""
        },
        {"role": "user", "content": pregunta}
    ]
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"❓ Pregunta: {pregunta}")
        print(f"{'='*50}")
    
    # Bucle del agente — puede usar múltiples tools en secuencia
    max_iteraciones = 5
    for iteracion in range(max_iteraciones):
        
        # El LLM decide si usar una tool o responder directamente
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=mensajes,
            tools=TOOLS,
            tool_choice="auto"  # el modelo decide cuándo usar tools
        )
        
        mensaje_respuesta = response.choices[0].message
        mensajes.append(mensaje_respuesta)
        
        # Si el modelo quiere usar una tool
        if mensaje_respuesta.tool_calls:
            for tool_call in mensaje_respuesta.tool_calls:
                nombre_tool = tool_call.function.name
                argumentos = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"\n🔧 Usando tool: {nombre_tool}")
                    print(f"   Argumentos: {argumentos}")
                
                # Ejecutar la tool
                if nombre_tool in FUNCIONES_DISPONIBLES:
                    resultado = FUNCIONES_DISPONIBLES[nombre_tool](**argumentos)
                else:
                    resultado = {"error": f"Tool {nombre_tool} no encontrada"}
                
                if verbose:
                    print(f"   Resultado: {json.dumps(resultado, ensure_ascii=False)[:150]}...")
                
                # Agregar resultado al historial de mensajes
                mensajes.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(resultado, ensure_ascii=False)
                })
        
        # Si el modelo da respuesta final (sin tools)
        else:
            respuesta_final = mensaje_respuesta.content
            if verbose:
                print(f"\n💬 Respuesta: {respuesta_final}")
            return respuesta_final
    
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