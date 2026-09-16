# Assistant IA interne — Expérimentation

## Contexte

Assistant IA interne fictif pour une PME éditrice de logiciel, permettant aux
employés de rechercher de l'information dans la documentation interne et
d'automatiser des tâches répétitives, via un LLM local (Ollama) avec function
calling.

## Stack

- Python 3.11 / FastAPI
- Ollama (Qwen2.5:7b) en local, API compatible OpenAI
- httpx (client async)
- pytest / pytest-httpx pour les tests

## Lancer le projet

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama serve  # si pas déjà lancé en service systemd
uvicorn src.main:app --reload
\`\`\`

Endpoint : `POST /chat` avec `{"message": "..."}`

## Lancer les tests

\`\`\`bash
pytest -v
\`\`\`
