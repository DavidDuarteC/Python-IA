# Python-IA

Proyecto de agentes IA inteligentes que pueden usar herramientas, razonar, y completar tareas autonomously.

## Prerrequisitos

- Python 3.12+
- Java 21+ (`brew install openjdk@21`)
- uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Agentespan (`uv pip install agentspan`)
- API key de algún provider (Groq, Gemini, HuggingFace, etc.)

## Setup

```bash
# Clonar repo
git clone <repo-url>
cd python-ia

# Instalar dependencias
uv sync

# Configurar API key en .env
echo "GROQ_API_KEY=tu_key" > .env

# Iniciar server (requiere Java 21)
export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"
source .env
agentspan server start
```

## Primeros Pasos

Tutoriales básicos para empezar:

- `Primeros pasos/agente_basico.py` - Tu primer agente con herramientas
- `Primeros pasos/prompt_engineering.py` - Mejores prácticas para prompts
- `Primeros pasos/rag_basico.py` - Retrieval Augmented Generation

```bash
uv run Primeros\ pasos/agente_basico.py
```

## Agents

- `agents/agent1.py`: Asistente personal con herramientas y memoria conversacional

## Providers Soportados

| Provider | Modelos | Gratis |
|----------|---------|--------|
| Groq | llama-3.1-8b, llama-3.1-70b | Sí (tier gratuito) |
| Gemini | gemini-2.0-flash, gemini-1.5-pro | Parcial |
| HuggingFace | Llama, Mistral | Sí (tier gratuito) |
| OpenAI | gpt-4o, gpt-4o-mini | No |

## Estructura

```
python-ia/
├── agents/           # Agentes personalizados
│   └── agent1.py
├── Primeros pasos/   # Tutoriales básicos
│   ├── agente_basico.py
│   ├── prompt_engineering.py
│   └── rag_basico.py
├── .env             # API keys (no commitear)
├── pyproject.toml   # Dependencias
├── uv.lock          # Versiones lockeadas
└── README.md
```

## Troubleshooting

**Error: Java not installed**
```bash
brew install openjdk@21
echo 'export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"' >> ~/.zshrc
```

**Server no responde**
```bash
agentspan server stop
agentspan server start
```

**Cuota agotada**: Esperar o usar otro provider
