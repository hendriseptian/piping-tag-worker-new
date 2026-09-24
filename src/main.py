import base64
import json
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from workers import asgi


APP_VERSION = "1.1.4"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.7-flash"

MAX_TILES_DEFAULT = 8
MAX_IMAGE_B64_DEFAULT = 5_500_000
MAX_TOTAL_IMAGE_B64_DEFAULT = 18_000_000

INTERACTIONS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)


app = FastAPI(
    title="Piping Tag Extractor Worker",
    version=APP_VERSION,
)

# GitHub Pages frontend -> Cloudflare Worker API.
#
# NOTE: FastAPI's CORSMiddleware is intentionally not used here.
# Cloudflare Python Workers uses an ASGI adapter, so we apply CORS at
# the ASGI boundary to guarantee that both preflight OPTIONS responses
# and normal API responses carry the required headers.
CORS_ORIGIN = "https://hendriseptian.github.io"


class ForceCORSMiddleware:
    """Small ASGI CORS layer compatible with Cloudflare Python Workers."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.application(scope, receive, send)
            return

        origin = ""
        for key, value in scope.get("headers", []):
            if key.lower() == b"origin":
                origin = value.decode("latin-1")
                break

        # Only enable CORS for the known GitHub Pages frontend.
        if origin != CORS_ORIGIN:
            await self.application(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()

        # Browser preflight for POST /api/extract.
        if method == "OPTIONS":
            headers = [
                (b"access-control-allow-origin", CORS_ORIGIN.encode()),
                (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                (b"access-control-allow-headers", b"Content-Type"),
                (b"access-control-max-age", b"86400"),
                (b"vary", b"Origin"),
            ]
            await send({
                "type": "http.response.start",
                "status": 204,
                "headers": headers,
            })
            await send({
                "type": "http.response.body",
                "body": b"",
            })
            return

        async def send_with_cors(message):
            if message.get("type") == "http.response.start":
                existing = {
                    key.lower() for key, _ in message.get("headers", [])
                }
                headers = list(message.get("headers", []))

                if b"access-control-allow-origin" not in existing:
                    headers.append(
                        (b"access-control-allow-origin", CORS_ORIGIN.encode())
                    )
                if b"vary" not in existing:
                    headers.append((b"vary", b"Origin"))

                message = dict(message)
                message["headers"] = headers

            await send(message)

        await self.application(scope, receive, send_with_cors)


# ============================================================
# JSON SCHEMAS
# ============================================================

TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "message": {"type": "string"},
    },
    "required": ["status", "message"],
}

OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "pid_no": {
            "type": "string",
            "description": "P&ID or drawing number only.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
        },
        "evidence": {
            "type": "string",
            "description": "Short visible transcription supporting the P&ID number.",
        },
    },
    "required": ["pid_no", "confidence", "evidence"],
}

TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag_no": {
                        "type": "string",
                        "description": "Exact visible piping line tag.",
                    },
                    "size_nps_in": {
                        "type": "string",
                        "description": "NPS in inches, derived from the line tag only.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Short exact transcription of the visible tag.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "tag_no",
                    "size_nps_in",
                    "evidence",
                    "confidence",
                ],
            },
        }
    },
    "required": ["tags"],
}


# ============================================================
# PROMPTS
# ============================================================

TEST_PROMPT = """
Return exactly:
status = ok
message = Gemini connection successful

Do not add anything else.
"""

OVERVIEW_PROMPT = r"""
You are reading one P&ID drawing image.

TASK:
Identify ONLY the P&ID / drawing number shown in the title block.

Look for a field explicitly associated with:
- P&ID No.
- P&ID Number
- Drawing No.
- Drawing Number

IMPORTANT:
- Do not return project numbers.
- Do not return document numbers.
- Do not return revision numbers.
- Do not return sheet/page numbers.
- Do not infer a number that is not readable.
- If the correct field cannot be read confidently, return an empty pid_no.
- Evidence must be a short transcription of the visible text.
"""

TAG_PROMPT = r"""
You are a specialist P&ID line-tag extraction system.

TASK:
Extract ONLY piping LINE TAGS visible in the supplied P&ID image tile.

A piping line tag is the designation printed for a process/utility piping line.

DO NOT extract:
- equipment tags
- valve tags
- instrument tags
- nozzle numbers
- dimensions
- elevations
- grid references
- revision numbers
- dates
- page numbers
- drawing numbers
- pressure/temperature values
- material specifications
- random numeric labels

ACCURACY RULES:

1. Transcribe the tag EXACTLY as visible.
2. Never invent, repair, autocorrect, or normalize characters.
3. Do not combine text from two different nearby labels.
4. Do not use a nearby valve/nozzle size as the line size.
5. Ignore a tag if it is cut off or genuinely ambiguous.
6. Prefer omission over guessing.
7. The evidence must be a short transcription of the visible line tag.
8. Confidence must be high, medium, or low.
9. Return only actual piping line tags.

NPS RULES:

- Extract NPS from the size explicitly belonging to the line tag.
- Example:
  1HC106-1"-FC2L
  => tag_no = 1HC106-1"-FC2L
  => size_nps_in = 1

- Example:
  1HC1-2"-FG2D
  => tag_no = 1HC1-2"-FG2D
  => size_nps_in = 2

- If the NPS is not explicitly readable in the tag, return an empty size_nps_in.
- Never guess the NPS from nearby graphics.

TILE BOUNDARY RULE:
If a tag is cut by the edge of the supplied tile, ignore it. It will be recovered from an overlapping neighboring tile.

Return JSON matching the supplied schema.
"""


# ============================================================
# ENVIRONMENT
# ============================================================

def env_value(request: Request, name: str, default: str) -> str:
    env = request.scope.get("env")
    if env is None:
        return default

    try:
        value = getattr(env, name)
        if value is not None and str(value).strip():
            return str(value)
    except Exception:
        pass

    return default


def env_int(request: Request, name: str, default: int) -> int:
    raw = env_value(request, name, str(default))
    try:
        return int(raw)
    except Exception:
        return default


# ============================================================
# CLOUDFLARE FETCH
# ============================================================

async def cf_fetch(url: str, options: dict[str, Any]):
    from js import Object, fetch
    from pyodide.ffi import to_js

    js_options = to_js(
        options,
        dict_converter=Object.fromEntries,
    )

    return await fetch(url, js_options)


# ============================================================
# GEMINI INTERACTIONS API
# ============================================================

def get_api_key(request: Request) -> str:
    env = request.scope.get("env")

    if env is None:
        return ""

    try:
        value = getattr(env, "GEMINI_API_KEY")
        return str(value or "").strip()
    except Exception:
        return ""


def parse_interaction_output(payload: dict[str, Any]) -> str:
    # Current Interactions API returns model output in steps.
    for step in payload.get("steps", []) or []:
        if not isinstance(step, dict):
            continue

        if step.get("type") != "model_output":
            continue

        for content in step.get("content", []) or []:
            if not isinstance(content, dict):
                continue

            if content.get("type") == "text" and content.get("text"):
                return str(content["text"])

    # Be tolerant if a future response exposes output_text directly.
    direct = payload.get("output_text")
    if direct:
        return str(direct)

    raise ValueError("Gemini returned no text output.")


def parse_json_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise ValueError("Gemini JSON output is not an object.")
        return value
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(cleaned[start:end + 1])
            if not isinstance(value, dict):
                raise ValueError("Gemini JSON output is not an object.")
            return value
        raise ValueError("Gemini returned invalid JSON.")


async def call_gemini(
    request: Request,
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    images: list[dict[str, str]] | None = None,
) -> dict[str, Any]:

    api_key = get_api_key(request)

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY secret is not configured."
        )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for image in images or []:
        content.append(
            {
                "type": "image",
                "data": image["data"],
                "mime_type": image["mime_type"],
            }
        )

    body = {
        "model": model,
        "store": False,
        "input": content,
        "response_format": [
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            }
        ],
    }

    response = await cf_fetch(
        INTERACTIONS_URL,
        {
            "method": "POST",
            "headers": {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
                "Api-Revision": "2026-05-20",
            },
            "body": json.dumps(body),
        },
    )

    status = int(response.status)
    raw = await response.text()

    if status < 200 or status >= 300:
        # Do not return the secret.
        raise RuntimeError(
            f"Gemini HTTP {status}: {raw[:1500]}"
        )

    payload = json.loads(raw)
    output_text = parse_interaction_output(payload)
    return parse_json_output(output_text)


# ============================================================
# IMAGE VALIDATION / NORMALIZATION
# ============================================================

def normalize_b64(value: str) -> str:
    value = str(value or "").strip()

    if value.startswith("data:"):
        comma = value.find(",")
        if comma < 0:
            raise ValueError("Invalid image data URL.")
        value = value[comma + 1:]

    return value


def validate_image(
    request: Request,
    image: Any,
    mime_type: Any,
) -> dict[str, str]:

    if not isinstance(image, str) or not image.strip():
        raise ValueError("Image data is missing.")

    data = normalize_b64(image)

    max_size = env_int(
        request,
        "MAX_IMAGE_B64",
        MAX_IMAGE_B64_DEFAULT,
    )

    if len(data) > max_size:
        raise ValueError(
            f"Image tile is too large. Limit is {max_size} base64 characters."
        )

    # Validate that the payload is actually base64.
    try:
        base64.b64decode(
            data,
            validate=True,
        )
    except Exception as exc:
        raise ValueError(
            "Image data is not valid base64."
        ) from exc

    mime = str(
        mime_type or "image/jpeg"
    ).strip().lower()

    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if mime not in allowed:
        raise ValueError(
            "Unsupported image MIME type. "
            "Use image/jpeg, image/png, or image/webp."
        )

    return {
        "data": data,
        "mime_type": mime,
    }


# ============================================================
# TAG NORMALIZATION
# ============================================================

def clean_tag(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    # Preserve the actual characters but remove accidental
    # whitespace introduced by OCR/vision transcription.
    value = re.sub(r"\s+", "", value)

    return value


def clean_size(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    value = (
        value
        .replace("″", '"')
        .replace("”", '"')
        .replace("“", '"')
    )

    value = re.sub(r"\s+", "", value)

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\"?",
        value,
    )

    if match:
        return match.group(1)

    return ""


def size_from_tag(tag: str) -> str:
    # Primary expected pattern:
    # 1HC106-1"-FC2L
    match = re.search(
        r'-(\d+(?:\.\d+)?)["″”](?:-|$)',
        tag,
    )

    if match:
        return match.group(1)

    # Secondary:
    # prefix-1"
    match = re.search(
        r'(\d+(?:\.\d+)?)["″”](?:-|$)',
        tag,
    )

    if match:
        return match.group(1)

    return ""


def dedupe_tags(
    items: list[dict[str, Any]],
    pid_no: str,
) -> list[dict[str, Any]]:

    seen: set[str] = set()
    result: list[dict[str, Any]] = []

    for item in items:
        tag = clean_tag(item.get("tag_no"))

        if not tag:
            continue

        key = tag.upper()

        if key in seen:
            continue

        seen.add(key)

        size = clean_size(
            item.get("size_nps_in")
        )

        if not size:
            size = size_from_tag(tag)

        result.append(
            {
                "tag_no": tag,
                "pid_no": pid_no,
                "from": "",
                "to": "",
                "size_nps_in": size,
                "evidence": str(
                    item.get("evidence") or ""
                ).strip(),
                "confidence": str(
                    item.get("confidence") or ""
                ).strip().lower(),
                "tile_id": str(
                    item.get("tile_id") or ""
                ),
            }
        )

    return result


# ============================================================
# ROUTES
# ============================================================

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
        "model": env_value(
            request,
            "GEMINI_MODEL",
            DEFAULT_MODEL,
        ),
        "fallback_model": env_value(
            request,
            "GEMINI_FALLBACK_MODEL",
            DEFAULT_FALLBACK_MODEL,
        ),
        "pipeline": (
            "Gemini Interactions API + high-resolution "
            "overlapping tile vision + strict JSON + deduplication"
        ),
        "output": [
            "Tag No.",
            "P&ID No.",
            "From",
            "To",
            "NPS (in)",
        ],
        "line_description": "blank",
    }


@app.get("/api/test-gemini")
@app.post("/api/test-gemini")
async def test_gemini(request: Request):
    model = env_value(
        request,
        "GEMINI_MODEL",
        DEFAULT_MODEL,
    )
    fallback = env_value(
        request,
        "GEMINI_FALLBACK_MODEL",
        DEFAULT_FALLBACK_MODEL,
    )

    try:
        result = await call_gemini(
            request,
            model=model,
            prompt=TEST_PROMPT,
            schema=TEST_SCHEMA,
        )

        return {
            "status": "ok",
            "gemini": "connected",
            "model": model,
            "fallback_model": fallback,
            "result": result,
        }

    except Exception as exc:
        # One fallback attempt.
        try:
            result = await call_gemini(
                request,
                model=fallback,
                prompt=TEST_PROMPT,
                schema=TEST_SCHEMA,
            )

            return {
                "status": "ok",
                "gemini": "connected",
                "model": fallback,
                "fallback_used": True,
                "result": result,
            }

        except Exception as fallback_exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Gemini connection failed.",
                    "primary_error": str(exc),
                    "fallback_error": str(fallback_exc),
                },
            )


@app.post("/api/extract")
async def extract(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body.",
        ) from exc

    tiles = body.get("tiles")
    overview = body.get("overview")

    if not isinstance(tiles, list) or not tiles:
        raise HTTPException(
            status_code=400,
            detail="tiles must be a non-empty array.",
        )

    max_tiles = env_int(
        request,
        "MAX_TILES",
        MAX_TILES_DEFAULT,
    )

    tiles = tiles[:max_tiles]

    normalized_tiles: list[dict[str, Any]] = []
    total_b64 = 0

    try:
        for index, tile in enumerate(tiles):
            if not isinstance(tile, dict):
                continue

            image = (
                tile.get("image")
                or tile.get("data")
                or tile.get("base64")
            )

            normalized = validate_image(
                request,
                image,
                tile.get("mime_type")
                or tile.get("mimeType")
                or "image/jpeg",
            )

            total_b64 += len(
                normalized["data"]
            )

            normalized_tiles.append(
                {
                    "id": str(
                        tile.get("id")
                        or index + 1
                    ),
                    **normalized,
                }
            )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    max_total = env_int(
        request,
        "MAX_TOTAL_IMAGE_B64",
        MAX_TOTAL_IMAGE_B64_DEFAULT,
    )

    if total_b64 > max_total:
        raise HTTPException(
            status_code=413,
            detail=(
                "Total image payload is too large. "
                "Reduce tile count or tile resolution."
            ),
        )

    if not normalized_tiles:
        raise HTTPException(
            status_code=400,
            detail="No valid image tiles were supplied.",
        )

    model = env_value(
        request,
        "GEMINI_MODEL",
        DEFAULT_MODEL,
    )

    fallback = env_value(
        request,
        "GEMINI_FALLBACK_MODEL",
        DEFAULT_FALLBACK_MODEL,
    )

    # --------------------------------------------------------
    # P&ID NUMBER
    # --------------------------------------------------------

    pid_no = ""
    pid_evidence = ""
    pid_confidence = ""

    if isinstance(overview, dict):
        overview_image = (
            overview.get("image")
            or overview.get("data")
            or overview.get("base64")
        )

        if overview_image:
            try:
                overview_img = validate_image(
                    request,
                    overview_image,
                    overview.get("mime_type")
                    or overview.get("mimeType")
                    or "image/jpeg",
                )

                overview_result = await call_gemini(
                    request,
                    model=model,
                    prompt=OVERVIEW_PROMPT,
                    schema=OVERVIEW_SCHEMA,
                    images=[overview_img],
                )

                pid_no = clean_tag(
                    overview_result.get("pid_no")
                )
                pid_evidence = str(
                    overview_result.get("evidence")
                    or ""
                ).strip()
                pid_confidence = str(
                    overview_result.get("confidence")
                    or ""
                ).strip().lower()

            except Exception:
                try:
                    overview_result = await call_gemini(
                        request,
                        model=fallback,
                        prompt=OVERVIEW_PROMPT,
                        schema=OVERVIEW_SCHEMA,
                        images=[overview_img],
                    )

                    pid_no = clean_tag(
                        overview_result.get("pid_no")
                    )
                    pid_evidence = str(
                        overview_result.get("evidence")
                        or ""
                    ).strip()
                    pid_confidence = str(
                        overview_result.get("confidence")
                        or ""
                    ).strip().lower()

                except Exception:
                    pass

    # --------------------------------------------------------
    # TILE EXTRACTION
    # --------------------------------------------------------

    async def process_tile(
        tile: dict[str, Any]
    ) -> list[dict[str, Any]]:

        try:
            result = await call_gemini(
                request,
                model=model,
                prompt=TAG_PROMPT,
                schema=TAG_SCHEMA,
                images=[
                    {
                        "data": tile["data"],
                        "mime_type": tile["mime_type"],
                    }
                ],
            )
        except Exception:
            result = await call_gemini(
                request,
                model=fallback,
                prompt=TAG_PROMPT,
                schema=TAG_SCHEMA,
                images=[
                    {
                        "data": tile["data"],
                        "mime_type": tile["mime_type"],
                    }
                ],
            )

        tags = result.get("tags", [])

        if not isinstance(tags, list):
            return []

        output = []

        for item in tags:
            if not isinstance(item, dict):
                continue

            tag = clean_tag(
                item.get("tag_no")
            )

            if not tag:
                continue

            size = clean_size(
                item.get("size_nps_in")
            )

            if not size:
                size = size_from_tag(tag)

            output.append(
                {
                    "tag_no": tag,
                    "size_nps_in": size,
                    "evidence": str(
                        item.get("evidence") or ""
                    ).strip(),
                    "confidence": str(
                        item.get("confidence") or ""
                    ).strip().lower(),
                    "tile_id": tile["id"],
                }
            )

        return output

    # Process tiles sequentially. This is intentionally conservative for
    # Cloudflare Workers: it avoids a burst of 8 simultaneous Gemini
    # requests and lets one failed tile fall back without killing the
    # entire extraction.
    raw_candidates: list[dict[str, Any]] = []
    tile_errors: list[dict[str, str]] = []

    for tile in normalized_tiles:
        try:
            items = await process_tile(tile)
            raw_candidates.extend(items)
        except Exception as exc:
            tile_errors.append(
                {
                    "tile_id": str(tile.get("id", "")),
                    "error": str(exc)[:1200],
                }
            )

    # If every tile failed, return a JSON error response rather than
    # allowing an unhandled exception to become a Cloudflare 500 page.
    if not raw_candidates and tile_errors:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Gemini could not process any image tile.",
                "tile_errors": tile_errors,
            },
        )

    final_tags = dedupe_tags(
        raw_candidates,
        pid_no,
    )

    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": model,
        "fallback_model": fallback,
        "pid_no": pid_no,
        "pid_evidence": pid_evidence,
        "pid_confidence": pid_confidence,
        "count": len(final_tags),
        "tags": final_tags,
        "meta": {
            "tiles_received": len(tiles),
            "tiles_processed": len(normalized_tiles),
            "raw_candidates": len(raw_candidates),
            "tile_errors": tile_errors,
            "line_description": "blank",
        },
    }


app = ForceCORSMiddleware(app)

Default = asgi.entrypoint(app)
