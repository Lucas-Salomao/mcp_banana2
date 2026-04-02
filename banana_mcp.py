"""
Servidor MCP para geração e edição de imagens com Nano Banana 2 (Gemini 3.1 Flash Image)
via Google AI Studio (Gemini API)

Instalação das dependências:
    pip install mcp google-genai pillow google-cloud-storage

Uso:
    Defina a variável de ambiente GEMINI_API_KEY com sua chave do Google AI Studio.
    Para upload no GCS, configure as credenciais do GCP:
        export GOOGLE_APPLICATION_CREDENTIALS="/caminho/para/service_account.json"
    Execute: python nano_banana_mcp_server.py
"""

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from google import genai
from google.genai import types

load_dotenv()


# ──────────────────────────────────────────────
# Modelos disponíveis
# ──────────────────────────────────────────────

MODELS = {
    "nano-banana-2":   "gemini-3.1-flash-image-preview",   # Rápido, alta qualidade, melhor custo-benefício
    "nano-banana-pro": "gemini-3-pro-image-preview",        # Máxima qualidade, texto preciso, raciocínio avançado
    "nano-banana":     "gemini-2.5-flash-image",            # Modelo base original
}

# ──────────────────────────────────────────────
# Inicialização
# ──────────────────────────────────────────────

app = Server("nano-banana-image-generator")


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Variável de ambiente GEMINI_API_KEY não definida. "
            "Obtenha sua chave em https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _resolve_model(model_name: str) -> str:
    """Resolve o alias amigável para o model ID real da API."""
    return MODELS.get(model_name, model_name)


def _load_image_inline(image_path: str) -> types.Part:
    """Carrega uma imagem do disco e retorna como inline Part."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    ext = path.suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    mime_type = mime_map.get(ext, "image/png")
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime_type)


def _save_response_images(response, output_path: str) -> list[str]:
    """Extrai e salva todas as imagens de uma resposta. Retorna lista de caminhos salvos."""
    saved = []
    base = Path(output_path)
    image_count = 0

    for part in response.parts:
        if part.inline_data is not None:
            mime = part.inline_data.mime_type or "image/png"
            ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
            ext = ext_map.get(mime, ".png")

            if image_count == 0:
                # Garante que o arquivo tenha a extensão correta
                out_file = base.with_suffix(ext)
            else:
                out_file = base.with_stem(f"{base.stem}_{image_count}").with_suffix(ext)

            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(part.inline_data.data)
            saved.append(str(out_file))
            image_count += 1

    return saved


# ──────────────────────────────────────────────
# Definição das ferramentas (tools)
# ──────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_image",
            description=(
                "Gera uma imagem a partir de um prompt de texto usando o Nano Banana 2 "
                "(Gemini 3.1 Flash Image). Suporta geração de alta qualidade com controle "
                "de aspect ratio. Use use_web_search ou use_image_search para aterrar a "
                "geração em informações reais da web."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Descrição detalhada da imagem a ser gerada.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Caminho para salvar a imagem gerada. Padrão: ./generated_image.png",
                        "default": "./generated_image.png",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"],
                        "description": (
                            "Modelo a usar. "
                            "nano-banana-2: rápido e de alta qualidade (recomendado). "
                            "nano-banana-pro: máxima qualidade, texto preciso. "
                            "nano-banana: modelo base."
                        ),
                        "default": "nano-banana-2-pro",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                        "description": (
                            "Proporção da imagem. Incluída no prompt para guiar o modelo. "
                            "Padrão: 1:1."
                        ),
                        "default": "1:1",
                    },
                    "use_web_search": {
                        "type": "boolean",
                        "description": (
                            "Ativa Google Web Search grounding antes de gerar a imagem. "
                            "Útil para locais reais, eventos, pessoas públicas etc. "
                            "Disponível apenas no nano-banana-2."
                        ),
                        "default": False,
                    },
                    "use_image_search": {
                        "type": "boolean",
                        "description": (
                            "Ativa Google Image Search grounding: o modelo busca imagens reais de referência. "
                            "Ex: animais, arquitetura, produtos específicos. "
                            "Disponível apenas no nano-banana-2."
                        ),
                        "default": False,
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Elementos que NÃO devem aparecer na imagem (opcional).",
                    },
                    "gcs_bucket_path": {
                        "type": "string",
                        "description": (
                            "Caminho do bucket GCS onde a imagem será salva (opcional). "
                            "Formatos: 'meu-bucket', 'meu-bucket/pasta/', 'gs://meu-bucket/pasta/img.png'. "
                            "Requer GOOGLE_APPLICATION_CREDENTIALS configurado. "
                            "O retorno incluirá o URI gs:// completo do arquivo salvo."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="edit_image",
            description=(
                "Edita uma imagem existente com base em instruções de texto usando o Nano Banana 2. "
                "Suporta: troca de fundo, mudança de estilo artístico, adição/remoção de objetos, "
                "colorização, transferência de estilo, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Instrução de edição. "
                            "Ex: 'Mude o fundo para uma praia tropical', "
                            "'Transforme em estilo anime', 'Adicione neve na cena'."
                        ),
                    },
                    "input_image_path": {
                        "type": "string",
                        "description": "Caminho da imagem a ser editada.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Caminho para salvar a imagem editada. Padrão: ./edited_image.png",
                        "default": "./edited_image.png",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"],
                        "default": "nano-banana-2-pro",
                    },
                    "mask_image_path": {
                        "type": "string",
                        "description": (
                            "Máscara de edição (imagem preto e branco). "
                            "Áreas brancas serão editadas, pretas preservadas (opcional)."
                        ),
                    },
                    "gcs_bucket_path": {
                        "type": "string",
                        "description": (
                            "Caminho do bucket GCS onde a imagem editada será salva (opcional). "
                            "Formatos: 'meu-bucket', 'meu-bucket/pasta/', 'gs://meu-bucket/pasta/img.png'. "
                            "Requer GOOGLE_APPLICATION_CREDENTIALS configurado."
                        ),
                    },
                },
                "required": ["prompt", "input_image_path"],
            },
        ),
        Tool(
            name="multi_image_edit",
            description=(
                "Combina ou edita múltiplas imagens (2-4) com instruções de texto. "
                "Útil para composição de cenas, transferência de estilo entre fotos, "
                "mesclar elementos de imagens diferentes, criar colagens, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Instrução descrevendo como combinar ou usar as imagens.",
                    },
                    "input_image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de 2 a 4 caminhos de imagens de entrada.",
                        "minItems": 2,
                        "maxItems": 4,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Caminho para salvar o resultado. Padrão: ./combined_image.png",
                        "default": "./combined_image.png",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"],
                        "default": "nano-banana-2-pro",
                    },
                    "gcs_bucket_path": {
                        "type": "string",
                        "description": (
                            "Caminho do bucket GCS onde a composição será salva (opcional). "
                            "Formatos: 'meu-bucket', 'meu-bucket/pasta/', 'gs://meu-bucket/pasta/img.png'. "
                            "Requer GOOGLE_APPLICATION_CREDENTIALS configurado."
                        ),
                    },
                },
                "required": ["prompt", "input_image_paths"],
            },
        ),
        Tool(
            name="describe_and_edit",
            description=(
                "Analisa uma imagem, gera uma descrição textual e aplica uma edição em uma única chamada. "
                "Útil para workflows conversacionais onde você quer saber o que o modelo vê "
                "antes de aplicar uma transformação."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "edit_instruction": {
                        "type": "string",
                        "description": (
                            "Instrução combinando descrição e edição. "
                            "Ex: 'Descreva esta cena e depois transforme em pintura a óleo'."
                        ),
                    },
                    "input_image_path": {
                        "type": "string",
                        "description": "Caminho da imagem de entrada.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Caminho para salvar a imagem resultante. Padrão: ./result_image.png",
                        "default": "./result_image.png",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["nano-banana-2", "nano-banana-pro", "nano-banana"],
                        "default": "nano-banana-2-pro",
                    },
                    "gcs_bucket_path": {
                        "type": "string",
                        "description": (
                            "Caminho do bucket GCS onde a imagem resultante será salva (opcional). "
                            "Formatos: 'meu-bucket', 'meu-bucket/pasta/', 'gs://meu-bucket/pasta/img.png'. "
                            "Requer GOOGLE_APPLICATION_CREDENTIALS configurado."
                        ),
                    },
                },
                "required": ["edit_instruction", "input_image_path"],
            },
        ),
    ]


# ──────────────────────────────────────────────
# Config padrão para geração de imagem
# A API do Nano Banana usa GenerateContentConfig com
# response_modalities=["IMAGE", "TEXT"] — sem ImageGenerationConfig.
# O aspect ratio é passado via texto no prompt.
# ──────────────────────────────────────────────

def _make_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )


def _upload_to_gcs(local_path: str, gcs_bucket_path: str) -> str:
    """
    Faz upload de um arquivo local para o Google Cloud Storage.

    Args:
        local_path:      Caminho local do arquivo (ex: /tmp/imagem.png)
        gcs_bucket_path: Destino no GCS. Aceita dois formatos:
                         - Apenas bucket:   "meu-bucket"
                         - Bucket + pasta:  "meu-bucket/pasta/" ou "meu-bucket/pasta/nome.png"
                         - Com prefixo gs:  "gs://meu-bucket/pasta/"

    Returns:
        URI completa no GCS: "gs://meu-bucket/pasta/imagem.png"
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        raise ImportError(
            "Pacote google-cloud-storage não instalado. "
            "Execute: pip install google-cloud-storage"
        )

    path_clean = gcs_bucket_path.removeprefix("gs://")

    if "/" in path_clean:
        bucket_name, blob_prefix = path_clean.split("/", 1)
        if blob_prefix.endswith("/") or "." not in Path(blob_prefix).name:
            blob_name = blob_prefix.rstrip("/") + "/" + Path(local_path).name
        else:
            blob_name = blob_prefix
    else:
        bucket_name = path_clean
        blob_name = Path(local_path).name

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Upload concluído: {gcs_uri}")
    return gcs_uri


# ──────────────────────────────────────────────
# Handlers das ferramentas
# ──────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    # ── generate_image ────────────────────────
    if name == "generate_image":
        prompt           = arguments["prompt"]
        output_path      = arguments.get("output_path", "./generated_image.png")
        model_name       = arguments.get("model", "nano-banana-2")
        aspect_ratio     = arguments.get("aspect_ratio", "1:1")
        use_web_search   = arguments.get("use_web_search", False)
        use_image_search = arguments.get("use_image_search", False)
        negative_prompt  = arguments.get("negative_prompt")
        gcs_bucket_path  = arguments.get("gcs_bucket_path")

        try:
            client = get_client()
            model_id = _resolve_model(model_name)

            # Incorpora aspect ratio e negative prompt diretamente no prompt
            full_prompt = f"{prompt}\n\nAspect ratio: {aspect_ratio}"
            if negative_prompt:
                full_prompt += f"\n\nNegative prompt (avoid these): {negative_prompt}"

            config = _make_config()

            # Ferramentas de busca (disponíveis apenas no nano-banana-2)
            tools = []
            if use_web_search or use_image_search:
                search_types = []
                if use_web_search:
                    search_types.append("web_search")
                if use_image_search:
                    search_types.append("image_search")
                tools.append(
                    types.Tool(google_search=types.GoogleSearch(search_types=search_types))
                )

            kwargs: dict = dict(model=model_id, contents=[full_prompt], config=config)
            if tools:
                kwargs["tools"] = tools

            response = client.models.generate_content(**kwargs)

            # Extrai texto descritivo (se houver)
            text_parts = [p.text for p in response.parts if p.text]
            description = "\n".join(text_parts) if text_parts else ""

            saved = _save_response_images(response, output_path)

            if not saved:
                return [TextContent(
                    type="text",
                    text=(
                        "⚠️ Nenhuma imagem foi retornada pela API. "
                        "O conteúdo pode ter sido bloqueado pelos filtros de segurança.\n"
                        + (f"Resposta do modelo: {description}" if description else "")
                    ),
                )]

            # Upload para GCS (se solicitado)
            gcs_uris = []
            if gcs_bucket_path:
                for local_file in saved:
                    gcs_uris.append(_upload_to_gcs(local_file, gcs_bucket_path))

            result = (
                f"✅ Imagem gerada com sucesso!\n"
                f"📁 Salva localmente em: {', '.join(saved)}\n"
                f"🎨 Modelo: {model_name} ({model_id})\n"
                f"📐 Aspect ratio: {aspect_ratio}\n"
            )
            if gcs_uris:
                result += f"☁️ URI no GCS: {', '.join(gcs_uris)}\n"
            if use_web_search:
                result += "🔍 Web Search grounding ativado\n"
            if use_image_search:
                result += "🖼️ Image Search grounding ativado\n"
            if description:
                result += f"\n📝 Comentário do modelo:\n{description}"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Erro ao gerar imagem: {str(e)}")]

    # ── edit_image ────────────────────────────
    elif name == "edit_image":
        prompt           = arguments["prompt"]
        input_image_path = arguments["input_image_path"]
        output_path      = arguments.get("output_path", "./edited_image.png")
        model_name       = arguments.get("model", "nano-banana-2")
        mask_image_path  = arguments.get("mask_image_path")
        gcs_bucket_path  = arguments.get("gcs_bucket_path")

        try:
            client = get_client()
            model_id = _resolve_model(model_name)

            image_part = _load_image_inline(input_image_path)
            contents = [prompt, image_part]
            if mask_image_path:
                contents.append(_load_image_inline(mask_image_path))

            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=_make_config(),
            )

            text_parts = [p.text for p in response.parts if p.text]
            description = "\n".join(text_parts) if text_parts else ""
            saved = _save_response_images(response, output_path)

            if not saved:
                return [TextContent(
                    type="text",
                    text=(
                        "⚠️ Nenhuma imagem editada foi retornada. "
                        "Verifique a instrução ou os filtros de segurança.\n"
                        + (f"Resposta: {description}" if description else "")
                    ),
                )]

            # Upload para GCS (se solicitado)
            gcs_uris = []
            if gcs_bucket_path:
                for local_file in saved:
                    gcs_uris.append(_upload_to_gcs(local_file, gcs_bucket_path))

            result = (
                f"✅ Imagem editada com sucesso!\n"
                f"📁 Salva localmente em: {', '.join(saved)}\n"
                f"🎨 Modelo: {model_name} ({model_id})\n"
            )
            if gcs_uris:
                result += f"☁️ URI no GCS: {', '.join(gcs_uris)}\n"
            if description:
                result += f"\n📝 Comentário do modelo:\n{description}"
            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Erro ao editar imagem: {str(e)}")]

    # ── multi_image_edit ──────────────────────
    elif name == "multi_image_edit":
        prompt            = arguments["prompt"]
        input_image_paths = arguments["input_image_paths"]
        output_path       = arguments.get("output_path", "./combined_image.png")
        model_name        = arguments.get("model", "nano-banana-2")
        gcs_bucket_path   = arguments.get("gcs_bucket_path")

        try:
            client = get_client()
            model_id = _resolve_model(model_name)

            contents = [prompt] + [_load_image_inline(p) for p in input_image_paths]

            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=_make_config(),
            )

            text_parts = [p.text for p in response.parts if p.text]
            description = "\n".join(text_parts) if text_parts else ""
            saved = _save_response_images(response, output_path)

            if not saved:
                return [TextContent(type="text", text=f"⚠️ Nenhuma imagem retornada.\n{description}")]

            # Upload para GCS (se solicitado)
            gcs_uris = []
            if gcs_bucket_path:
                for local_file in saved:
                    gcs_uris.append(_upload_to_gcs(local_file, gcs_bucket_path))

            result = (
                f"✅ Composição de imagens concluída!\n"
                f"📁 Salva localmente em: {', '.join(saved)}\n"
                f"🎨 Modelo: {model_name} ({model_id})\n"
                f"🖼️ {len(input_image_paths)} imagens de entrada usadas\n"
            )
            if gcs_uris:
                result += f"☁️ URI no GCS: {', '.join(gcs_uris)}\n"
            if description:
                result += f"\n📝 Comentário do modelo:\n{description}"
            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Erro na composição: {str(e)}")]

    # ── describe_and_edit ─────────────────────
    elif name == "describe_and_edit":
        instruction      = arguments["edit_instruction"]
        input_image_path = arguments["input_image_path"]
        output_path      = arguments.get("output_path", "./result_image.png")
        model_name       = arguments.get("model", "nano-banana-2")
        gcs_bucket_path  = arguments.get("gcs_bucket_path")

        try:
            client = get_client()
            model_id = _resolve_model(model_name)

            image_part = _load_image_inline(input_image_path)

            response = client.models.generate_content(
                model=model_id,
                contents=[instruction, image_part],
                config=_make_config(),
            )

            text_parts = [p.text for p in response.parts if p.text]
            description = "\n".join(text_parts) if text_parts else ""
            saved = _save_response_images(response, output_path)

            # Upload para GCS (se solicitado)
            gcs_uris = []
            if gcs_bucket_path and saved:
                for local_file in saved:
                    gcs_uris.append(_upload_to_gcs(local_file, gcs_bucket_path))

            result = f"🎨 Modelo: {model_name} ({model_id})\n"
            if description:
                result += f"\n📝 Análise e resposta do modelo:\n{description}\n"
            if saved:
                result += f"\n✅ Imagem resultante salva localmente em: {', '.join(saved)}"
            else:
                result += "\n⚠️ Nenhuma imagem foi gerada (resposta apenas em texto)."
            if gcs_uris:
                result += f"\n☁️ URI no GCS: {', '.join(gcs_uris)}"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Erro: {str(e)}")]

    return [TextContent(type="text", text=f"❌ Ferramenta desconhecida: {name}")]


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())