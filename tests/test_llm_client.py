import pytest
from llm_client import chat_send, OllamaError

@pytest.mark.asyncio
async def test_chat_send_reponse_normale(httpx_mock):
    # On simule une réponse Ollama classique, sans tool_calls
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        json={
            "choices": [{
                "message": {"role": "assistant", "content": "Bonjour !"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
    )
    result = await chat_send([{"role": "user", "content": "salut"}])
    assert result["message"]["content"] == "Bonjour !"
    assert result["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_send_ollama_down(httpx_mock):
    # On simule Ollama injoignable (ConnectError)
    import httpx
    httpx_mock.add_exception(httpx.ConnectError("connexion refusée"))

    with pytest.raises(OllamaError):
        await chat_send([{"role": "user", "content": "salut"}])

        
@pytest.mark.asyncio
async def test_chat_send_avec_tool_call(httpx_mock):
    # 1er appel : le modèle décide d'appeler un outil
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        json={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_test123",
                        "type": "function",
                        "function": {
                            "name": "chercher_documentation",
                            "arguments": '{"query": "ventes du mois"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }
    )

    result = await chat_send(
        [{"role": "user", "content": "combien on a vendu ce mois-ci"}],
        tools=[]  # peu importe le contenu réel ici, seul le fait d'en envoyer compte
    )

    assert result["finish_reason"] == "tool_calls"
    assert result["message"]["tool_calls"][0]["function"]["name"] == "chercher_documentation"