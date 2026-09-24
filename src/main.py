import asyncio
import base64
import json
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from workers import asgi


APP_VERSION = "1.0.0"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK = "gemini-3.7-flash"
MAX_TILES_DEFAULT = 8
MAX_IMAGE_B64 = 5_500_000
MAX_TOTAL_IMAGE_B64 = 18_000_000

app = FastAPI(
    title="Piping Tag Extractor Worker",
    version=APP_VERSION,
)


TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag_no": {"type": "string"},
                    "size_nps_in": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "string"},
                },
                "required": ["tag_no", "size_nps_in", "evidence", "confidence"],
            },
        }
    },
    "required": ["tags"],
}


OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "pid_no": {"type": "string"},
        "confidence": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["pid_no", "confidence", "evidence"],
}


EXTRACT_PROMPT = r"""
You are a specialist P&ID document extraction system.

TASK:
Extract ONLY piping LINE TAGS visible in this image.

A piping tag is the alphanumeric line designation printed on a P&ID near a pipe.
Do NOT extract:
- equipment tags
- valve tags
- instrument tags
- nozzle numbers
- drawing dimensions
- elevation numbers
- grid references
- page numbers
- revision numbers
- material specifications
- pressure/temperature values
- text from the title block unless it is clearly a piping line tag

IMPORTANT:
1. Read the tag exactly as printed.
2. Do not invent missing characters.
3. Do not normalize a tag into a format that is not visibly supported.
4. Ignore partial/cut-off tags.
5. If a candidate is ambiguous, DO NOT include it.
6. A tag may contain a size such as 1", 2", 3", 4", etc.
7. Extract the NPS size ONLY when it is explicitly part of the tag or unambiguously printed as the tag's line-size field.
8. The output must contain only tags that are actually visible.
9. Keep duplicate visible occurrences in the raw response; the server will deduplicate later.
10. Evidence must be a short exact visual transcription showing why the tag was accepted.

Return JSON matching the supplied schema.
"""

OVERVIEW_PROMPT = r"""
You are reading a P&ID drawing overview.

TASK:
Identify the P&ID / drawing number shown in the title block.

Rules:
- Read only the drawing/P&ID number.
- Do not return project number, client number, sheet number, revision, date, or document number unless it is clearly the field labeled P&ID No. / Drawing No.
- If the field cannot be read confidently, return an empty string.
- Do not guess.
- Evidence must be the exact visible text used to make the decision.

Return JSON matching the supplied schema.
"""


def env_value(request: Request, name: str, default: str) -> str:
    env = request.scope.get("env")
    if env is not None:
        try:
            value = getattr(env, name)
            if value:
                return str(value)
        except Exception:
            pass
    return default


def normalize_b64(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("image data must be a string")
    value = value.strip()
    if value.startswith("data:"):
        comma = value.find(",")
        if comma < 0:
            raise ValueError("invalid data URL")
        value = value[comma + 1:]
    return value


def image_part(image_b64: str, mime_type: str) -> dict[str, Any]:
    data = normalize_b64(image_b64)
    if len(data) > MAX_IMAGE_B64:
        raise ValueError("image tile is too large")
    return {
        "inline_data": {
            "mime_type": mime_type or "image/jpeg",
            "data": data,
        }
    }


def extract_json_from_response(payload: dict[str, Any]) -> Any:
    # generateContent response
    candidates = payload.get("candidates") or []
    if candidates:
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        text_parts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
        if text_parts:
            return parse_json_text("\n".join(text_parts))

    # Interactions-style response, for future compatibility
    output = payload.get("output") or payload.get("steps") or []
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        texts.append(str(block["text"]))
        if texts:
            return parse_json_text("\n".join(texts))

    raise ValueError("Gemini returned no text output")


def parse_json_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


async def call_gemini(
    request: Request,
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    image_b64: str | None = None,
    mime_type: str = "image/jpeg",
) -> Any:
    env = request.scope.get("env")
    if env is None:
        raise RuntimeError("Cloudflare env is unavailable")

    try:
        api_key = str(getattr(env, "GEMINI_API_KEY"))
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY secret is not configured")

    parts: list[dict[str, Any]] = []
    if image_b64:
        parts.append(image_part(image_b64, mime_type))
    parts.append({"text": prompt})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent"
    )

    response = await request_js_fetch(
        url,
        {
            "method": "POST",
            "headers": {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            "body": json.dumps(body),
        },
    )

    status = int(response.status)
    raw = await response.text()

    if status < 200 or status >= 300:
        raise RuntimeError(f"Gemini HTTP {status}: {raw[:1000]}")

    payload = json.loads(raw)
    return extract_json_from_response(payload)


async def request_js_fetch(url: str, options: dict[str, Any]):
    # Cloudflare Python Workers exposes the Fetch API through the JS FFI.
    from js import Object, fetch
    from pyodide.ffi import to_js

    js_options = to_js(options, dict_converter=Object.fromEntries)
    return await fetch(url, js_options)


def clean_tag(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", "", value)
    return value


def clean_size(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    value = value.replace("″", '"').replace("”", '"').replace("“", '"')
    value = re.sub(r"\s+", "", value)
    m = re.search(r"(\d+(?:\.\d+)?)", value)
    return m.group(1) if m else ""


def dedupe_tags(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []

    for item in items:
        tag = clean_tag(item.get("tag_no"))
        size = clean_size(item.get("size_nps_in"))
        if not tag:
            continue

        key = tag.upper()
        if key in seen:
            continue

        seen.add(key)
        result.append(
            {
                "tag_no": tag,
                "pid_no": "",
                "from": "",
                "to": "",
                "size_nps_in": size,
                "evidence": str(item.get("evidence") or "").strip(),
                "confidence": str(item.get("confidence") or "").strip().lower(),
            }
        )

    return result


def parse_size_from_tag(tag: str) -> str:
    # Typical line tag example: 1HC106-1"-FC2L
    m = re.search(r'-(\d+(?:\.\d+)?)["″”]', tag)
    if m:
        return m.group(1)

    # Conservative fallback: a size-like number immediately followed by inch mark.
    m = re.search(r'(?:^|-)(\d+(?:\.\d+)?)["″”](?:-|$)', tag)
    return m.group(1) if m else ""


@app.get("/")
async def root():
    return {
        "name": "Piping Tag Extractor Worker",
        "version": APP_VERSION,
        "status": "ok",
    }


@app.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "service": "piping-tag-extractor",
        "version": APP_VERSION,
        "model": env_value(request, "GEMINI_MODEL", DEFAULT_MODEL),
        "fallback_model": env_value(request, "GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK),
        "pipeline": "overview + high-resolution tile vision + strict JSON + deduplication",
        "output": ["Tag No.", "P&ID No.", "From", "To", "NPS (in)"],
        "line_description": "blank",
    }


@app.post("/api/extract")
async def extract(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    overview = body.get("overview")
    tiles = body.get("tiles")

    if not isinstance(tiles, list) or not tiles:
        raise HTTPException(status_code=400, detail="tiles must be a non-empty array")

    max_tiles = int(env_value(request, "MAX_TILES", str(MAX_TILES_DEFAULT)))
    tiles = tiles[:max_tiles]

    total = 0
    normalized_tiles: list[dict[str, str]] = []

    for idx, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            continue
        image = tile.get("image") or tile.get("data") or tile.get("base64")
        if not image:
            continue
        mime = str(tile.get("mime_type") or tile.get("mimeType") or "image/jpeg")
        data = normalize_b64(str(image))
        total += len(data)
        if len(data) > MAX_IMAGE_B64:
            raise HTTPException(status_code=413, detail=f"Tile {idx + 1} is too large")
        normalized_tiles.append(
            {
                "id": str(tile.get("id") or idx + 1),
                "image": data,
                "mime_type": mime,
            }
        )

    if not normalized_tiles:
        raise HTTPException(status_code=400, detail="No valid image tiles supplied")

    if total > MAX_TOTAL_IMAGE_B64:
        raise HTTPException(
            status_code=413,
            detail="Total tile image data is too large; send fewer/smaller tiles",
        )

    model = env_value(request, "GEMINI_MODEL", DEFAULT_MODEL)
    fallback = env_value(request, "GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK)

    pid_no = ""
    pid_evidence = ""
    pid_confidence = ""

    if isinstance(overview, dict):
        overview_image = overview.get("image") or overview.get("data") or overview.get("base64")
        if overview_image:
            try:
                result = await call_gemini(
                    request,
                    model=model,
                    prompt=OVERVIEW_PROMPT,
                    schema=OVERVIEW_SCHEMA,
                    image_b64=normalize_b64(str(overview_image)),
                    mime_type=str(overview.get("mime_type") or "image/jpeg"),
                )
                pid_no = clean_tag(result.get("pid_no"))
                pid_evidence = str(result.get("evidence") or "")
                pid_confidence = str(result.get("confidence") or "")
            except Exception:
                try:
                    result = await call_gemini(
                        request,
                        model=fallback,
                        prompt=OVERVIEW_PROMPT,
                        schema=OVERVIEW_SCHEMA,
                        image_b64=normalize_b64(str(overview_image)),
                        mime_type=str(overview.get("mime_type") or "image/jpeg"),
                    )
                    pid_no = clean_tag(result.get("pid_no"))
                    pid_evidence = str(result.get("evidence") or "")
                    pid_confidence = str(result.get("confidence") or "")
                except Exception:
                    pass

    async def process_tile(tile: dict[str, str]) -> list[dict[str, Any]]:
        try:
            result = await call_gemini(
                request,
                model=model,
                prompt=EXTRACT_PROMPT,
                schema=TAG_SCHEMA,
                image_b64=tile["image"],
                mime_type=tile["mime_type"],
            )
        except Exception:
            result = await call_gemini(
                request,
                model=fallback,
                prompt=EXTRACT_PROMPT,
                schema=TAG_SCHEMA,
                image_b64=tile["image"],
                mime_type=tile["mime_type"],
            )

        tags = result.get("tags", [])
        if not isinstance(tags, list):
            return []

        cleaned: list[dict[str, Any]] = []
        for item in tags:
            if not isinstance(item, dict):
                continue
            tag = clean_tag(item.get("tag_no"))
            if not tag:
                continue
            size = clean_size(item.get("size_nps_in"))
            if not size:
                size = parse_size_from_tag(tag)

            cleaned.append(
                {
                    "tag_no": tag,
                    "size_nps_in": size,
                    "evidence": item.get("evidence", ""),
                    "confidence": item.get("confidence", ""),
                    "tile_id": tile["id"],
                }
            )
        return cleaned

    # Small concurrency avoids hammering the API while keeping the Worker responsive.
    results = await asyncio.gather(*(process_tile(tile) for tile in normalized_tiles))

    raw_tags: list[dict[str, Any]] = []
    for tile_items in results:
        raw_tags.extend(tile_items)

    tags = dedupe_tags(raw_tags)

    for item in tags:
        item["pid_no"] = pid_no

    return {
        "status": "ok",
        "model": model,
        "fallback_model": fallback,
        "pid_no": pid_no,
        "pid_evidence": pid_evidence,
        "pid_confidence": pid_confidence,
        "count": len(tags),
        "tags": tags,
        "meta": {
            "tiles_processed": len(normalized_tiles),
            "raw_candidates": len(raw_tags),
            "line_description": "blank",
        },
    }


Default = asgi.entrypoint(app)
