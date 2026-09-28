from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tools import TOOLS_SCHEMA, AVAILABLE_TOOLS
from llm_client import OllamaError
from llm_client import chat_send
from llm_client import chat_stream
import json

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatReply(BaseModel):
    reply: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    messages = [{"role": "user", "content": request.message}]
    try:
        choice = await chat_send(messages, tools=TOOLS_SCHEMA)

        if choice["finish_reason"] == "tool_calls":
            tool_calls = choice["message"]["tool_calls"]

            # on ajoute le message assistant original à l'historique
            messages.append(choice["message"])

            # on exécute chaque tool demandé
            for tool_call in tool_calls:
                nom_tool = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])  # string -> dict

                fonction = AVAILABLE_TOOLS[nom_tool]
                resultat = fonction(**arguments) 

                # on ajoute le résultat de l'outil à l'historique, lié à l'id du tool_call
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": resultat,
                })

            # 2e appel : le modèle reformule sa réponse finale avec le résultat sous les yeux
            choice = await chat_send(messages)  # pas de tools ici, on veut du texte final
            content = choice["message"]["content"]
        else:
            content = choice["message"]["content"]

        return ChatReply(reply=content)
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    messages = [{"role": "user", "content": request.message}]

    async def event_generator():
        tool_calls_result = None

        async for event in chat_stream(messages, tools=TOOLS_SCHEMA):
            if event["type"] == "content":
                yield f"data: {json.dumps({'content': event['data']})}\n\n"
            elif event["type"] == "tool_calls":
                tool_calls_result = event["data"]

        if tool_calls_result:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls_result
                ],
            })
            for tc in tool_calls_result:
                resultat = AVAILABLE_TOOLS[tc["name"]](**json.loads(tc["arguments"]))
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": resultat})

            # 2e appel, en streaming, sans tools : c'est la réponse finale
            async for event in chat_stream(messages):
                if event["type"] == "content":
                    yield f"data: {json.dumps({'content': event['data']})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")