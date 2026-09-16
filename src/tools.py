from llm_client import chat_send

# tools.py

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "chercher_documentation",
            "description": "Recherche une information dans la documentation interne de l'entreprise. Utilise cet outil pour toute question métier, technique ou de process interne.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Les mots-clés ou la question à rechercher"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def chercher_documentation(query: str) -> str:
    # Mock
    return (
        "D'après la documentation interne : les ventes du mois en cours "
        "s'élèvent à 42 500€, en hausse de 12% par rapport au mois précédent. "
        "Source : rapport-ventes-mensuel.md"
    )

# Registre nom → fonction, pour appeler dynamiquement selon ce que le modèle demande
AVAILABLE_TOOLS = {
    "chercher_documentation": chercher_documentation,
}