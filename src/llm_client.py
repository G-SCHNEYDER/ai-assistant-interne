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

async def chat_stream(messages: list[dict], model: str = "qwen2.5:7b", tools: list = None):
    """Générateur async : yield des événements {"type": "content"|"tool_calls", "data": ...}."""
    payload = {"model": model, "messages": messages, "stream": True}
    if tools:
        payload["tools"] = tools

    tool_call_buffer = {}  # index -> {"id", "name", "arguments"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                    if line == "[DONE]":
                        break

                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"]

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            buf = tool_call_buffer.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            if tc.get("id"):
                                buf["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                buf["name"] += fn["name"]
                            if fn.get("arguments"):
                                buf["arguments"] += fn["arguments"]
                    elif delta.get("content"):
                        yield {"type": "content", "data": delta["content"]}

        # Après la boucle : on émet le tool call assemblé, sans dépendre de finish_reason
        if tool_call_buffer:
            yield {"type": "tool_calls", "data": list(tool_call_buffer.values())}

    except httpx.ConnectError as e:
        raise OllamaError("Impossible de se connecter à Ollama (serveur éteint ?)") from e
    except httpx.TimeoutException as e:
        raise OllamaError("Ollama a mis trop de temps à répondre (timeout)") from e
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama a renvoyé une erreur HTTP {e.response.status_code}") from e