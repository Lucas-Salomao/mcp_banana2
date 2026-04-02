"""
Servidor MCP para geração e edição de imagens com Nano Banana 2 (Gemini 3.1 Flash Image)
via Google AI Studio (Gemini API) com transporte SSE (HTTP) para deploy remoto (GKE).

Versão ASGI Pura: Máxima estabilidade para conexões SSE e HTTP remotos.
Inclui metadados completos e suporte a Resolução (1K, 2K, 4K).
"""

import os
import asyncio
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from google import genai
from google.genai import types

load_dotenv()

# ──────────────────────────────────────────────
# Configuração e Modelos
# ──────────────────────────────────────────────

PORT = int(os.getenv("PORT", "8080"))

MODELS = {
    "nano-banana-2":   "gemini-3.1-flash-image-preview",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "nano-banana":     "gemini-2.5-flash-image",
}

app_mcp = Server("nano-banana-image-generator")

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não definida. Obtenha em https://aistudio.google.com/apikey")
    return genai.Client(api_key=api_key)

def _resolve_model(model_name: str) -> str:
    return MODELS.get(model_name, model_name)

def _load_image_resource(image_path: str) -> types.Part:
    if image_path.startswith("gs://"):
        ext = Path(image_path).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        return types.Part.from_uri(file_uri=image_path, mime_type=mime_map.get(ext, "image/png"))
    
    path = Path(image_path)
    if not path.exists(): raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime_map.get(ext, "image/png"))

def _save_response_images(response, output_path: str) -> list[str]:
    saved = []
    base = Path(output_path)
    image_count = 0
    for part in response.parts:
        if part.inline_data is not None:
            mime = part.inline_data.mime_type or "image/png"
            ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
            ext = ext_map.get(mime, ".png")
            out_file = base.with_suffix(ext) if image_count == 0 else base.with_stem(f"{base.stem}_{image_count}").with_suffix(ext)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(part.inline_data.data)
            saved.append(str(out_file))
            image_count += 1
    return saved

def _upload_to_gcs(local_path: str, gcs_bucket_path: str) -> str:
    from google.cloud import storage
    path_clean = gcs_bucket_path.removeprefix("gs://")
    if "/" in path_clean:
        bucket_name, blob_prefix = path_clean.split("/", 1)
        blob_name = blob_prefix.rstrip("/") + "/" + Path(local_path).name if (blob_prefix.endswith("/") or "." not in Path(blob_prefix).name) else blob_prefix
    else:
        bucket_name = path_clean
        blob_name = Path(local_path).name
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"

def _make_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

# ──────────────────────────────────────────────
# Schema de Ferramentas
# ──────────────────────────────────────────────

RESOLUTION_PROPERTY = {
    "type": "string",
    "enum": ["1K", "2K", "4K"],
    "description": "Resolução da imagem (apenas para Pro/Flash). Padrão: 1K.",
    "default": "1K"
}

@app_mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_image",
            description="Gera uma imagem a partir de um prompt de texto. Suporta Pro e Flash com opções de Resolução e Aspect Ratio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Descrição detalhada da imagem."},
                    "output_path": {"type": "string", "default": "./generated_image.png"},
                    "model": {"type": "string", "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"], "default": "nano-banana-2"},
                    "resolution": RESOLUTION_PROPERTY,
                    "aspect_ratio": {"type": "string", "enum": ["1:1", "3:4", "4:3", "9:16", "16:9"], "default": "1:1"},
                    "use_web_search": {"type": "boolean", "default": False},
                    "use_image_search": {"type": "boolean", "default": False},
                    "negative_prompt": {"type": "string"},
                    "gcs_bucket_path": {"type": "string"}
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="edit_image",
            description="Edita uma imagem original. Suporta GCS URIs e Resolução.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Instrução de edição."},
                    "input_image_path": {"type": "string", "description": "Caminho local ou gs:// URI."},
                    "output_path": {"type": "string", "default": "./edited_image.png"},
                    "model": {"type": "string", "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"], "default": "nano-banana-2"},
                    "resolution": RESOLUTION_PROPERTY,
                    "mask_image_path": {"type": "string"},
                    "gcs_bucket_path": {"type": "string"}
                },
                "required": ["prompt", "input_image_path"]
            }
        ),
        Tool(
            name="multi_image_edit",
            description="Combina múltiplas imagens (2-4) com suporte a Resolução.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "input_image_paths": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
                    "output_path": {"type": "string", "default": "./combined_image.png"},
                    "model": {"type": "string", "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"], "default": "nano-banana-2"},
                    "resolution": RESOLUTION_PROPERTY,
                    "gcs_bucket_path": {"type": "string"}
                },
                "required": ["prompt", "input_image_paths"]
            }
        ),
        Tool(
            name="describe_and_edit",
            description="Análise e edição em chamada única com suporte a Resolução.",
            inputSchema={
                "type": "object",
                "properties": {
                    "edit_instruction": {"type": "string"},
                    "input_image_path": {"type": "string"},
                    "output_path": {"type": "string", "default": "./result_image.png"},
                    "model": {"type": "string", "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"], "default": "nano-banana-2"},
                    "resolution": RESOLUTION_PROPERTY,
                    "gcs_bucket_path": {"type": "string"}
                },
                "required": ["edit_instruction", "input_image_path"]
            }
        )
    ]

# ──────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────

@app_mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    print(f"DEBUG: Chamando ferramenta {name} com argumentos {arguments}")
    client = get_client()
    model_name = arguments.get("model", "nano-banana-2")
    model_id = _resolve_model(model_name)
    gcs_bucket_path = arguments.get("gcs_bucket_path")
    resolution = arguments.get("resolution", "1K")

    try:
        res_suffix = f"\nOutput resolution: {resolution}" if model_name in ["nano-banana-2", "nano-banana-pro"] else ""

        if name == "generate_image":
            prompt = arguments["prompt"]
            output_path = arguments.get("output_path", "./generated_image.png")
            aspect_ratio = arguments.get("aspect_ratio", "1:1")
            use_web_search = arguments.get("use_web_search", False)
            use_image_search = arguments.get("use_image_search", False)
            negative_prompt = arguments.get("negative_prompt")

            full_prompt = f"{prompt}\n\nAspect ratio: {aspect_ratio}{res_suffix}"
            if negative_prompt: full_prompt += f"\n\nNegative prompt: {negative_prompt}"

            config = _make_config()
            if use_web_search or use_image_search:
                search_types = [t for t, enabled in [("web_search", use_web_search), ("image_search", use_image_search)] if enabled]
                config.tools = [types.Tool(google_search=types.GoogleSearch(search_types=search_types))]

            response = client.models.generate_content(model=model_id, contents=[full_prompt], config=config)

        elif name == "edit_image":
            prompt = arguments["prompt"] + res_suffix
            input_path = arguments["input_image_path"]
            output_path = arguments.get("output_path", "./edited_image.png")
            mask_path = arguments.get("mask_image_path")

            contents = [prompt, _load_image_resource(input_path)]
            if mask_path: contents.append(_load_image_resource(mask_path))

            response = client.models.generate_content(model=model_id, contents=contents, config=_make_config())

        elif name == "multi_image_edit":
            prompt = arguments["prompt"] + res_suffix
            input_paths = arguments["input_image_paths"]
            output_path = arguments.get("output_path", "./combined_image.png")

            contents = [prompt] + [_load_image_resource(p) for p in input_paths]
            response = client.models.generate_content(model=model_id, contents=contents, config=_make_config())

        elif name == "describe_and_edit":
            instruction = arguments["edit_instruction"] + res_suffix
            input_path = arguments["input_image_path"]
            output_path = arguments.get("output_path", "./result_image.png")

            response = client.models.generate_content(model=model_id, contents=[instruction, _load_image_resource(input_path)], config=_make_config())
        else:
            return [TextContent(type="text", text=f"❌ Ferramenta desconhecida: {name}")]

        text_parts = [p.text for p in response.parts if p.text]
        description = "\n".join(text_parts) if text_parts else ""
        saved = _save_response_images(response, output_path)

        gcs_uris = []
        if gcs_bucket_path and saved:
            gcs_uris = [_upload_to_gcs(f, gcs_bucket_path) for f in saved]

        result = f"✅ Sucesso!\n📁 Local: {', '.join(saved)}\n🎨 Modelo: {model_name}\n📏 Resolução: {resolution if model_name != 'nano-banana' else 'Standard (Lite)'}"
        if gcs_uris: result += f"\n☁️ GCS: {', '.join(gcs_uris)}"
        if description: result += f"\n\n📝 Comentário:\n{description}"
        return [TextContent(type="text", text=result)]

    except Exception as e:
        print(f"ERRO: {e}")
        return [TextContent(type="text", text=f"❌ Erro: {str(e)}")]

# ──────────────────────────────────────────────
# Lógica do Servidor ASGI DIRETA
# ──────────────────────────────────────────────

sse = SseServerTransport("/messages")

async def app(scope, receive, send):
    """Aplicação ASGI Pura compatível com Uvicorn e GKE."""
    
    # 1. Tratamento de CORS manual (Simples e Direto)
    if scope["type"] == "http":
        headers = dict(scope.get("headers", []))
        if scope["method"] == "OPTIONS":
            await send({
                "type": "http.response.start", "status": 204,
                "headers": [
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"*"),
                    (b"access-control-allow-headers", b"*"),
                ]
            })
            await send({"type": "http.response.body", "body": b""})
            return

    # 2. Roteamento de Rotas MCP
    if scope["type"] == "http":
        path = scope["path"]
        
        # Conexão SSE Principal
        if path == "/sse":
            print("DEBUG: Iniciando conexão SSE...")
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await app_mcp.run(read_stream, write_stream, app_mcp.create_initialization_options())
            return
            
        # Mensagens POST (JSON-RPC)
        elif path.startswith("/messages"):
            print("DEBUG: Recebendo POST em /messages")
            await sse.handle_post_message(scope, receive, send)
            return

    # 3. Fallback 404
    if scope["type"] == "http":
        await send({
            "type": "http.response.start", "status": 404,
            "headers": [(b"content-type", b"text/plain")]
        })
        await send({"type": "http.response.body", "body": b"MCP Server Running. Use /sse to connect."})

if __name__ == "__main__":
    import uvicorn
    print(f"Banana MCP SSE (Pure ASGI) na porta {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
