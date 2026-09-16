import httpx
import time
import logging
import json
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

class OllamaError(Exception):
    """Erreur levée quand l'appel à Ollama échoue."""
    pass

async def chat_send(messages: list[dict], model: str = "qwen2.5:7b", tools: list = None) -> dict:    
    try:
        payload={"model": model, "messages": messages}

        if tools:
            payload["tools"]=tools

        debut = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_URL,
                json=payload,
                timeout=60
            )
            duree_ms = (time.time() - debut) * 1000
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            tokens_prompt = usage.get("prompt_tokens")
            tokens_completion = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "latence_ms": duree_ms,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "total_tokens": total_tokens
            }
            logger.info(json.dumps(log_data))
            return data["choices"][0]
    except httpx.ConnectError as e:
        raise OllamaError("Impossible de se connecter à Ollama (serveur éteint ?)") from e

    except httpx.TimeoutException as e:
        raise OllamaError("Ollama a mis trop de temps à répondre (timeout)") from e

    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama a renvoyé une erreur HTTP {e.response.status_code}") from e