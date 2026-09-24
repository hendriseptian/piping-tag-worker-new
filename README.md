# Piping Tag Extractor Worker

Dedicated Cloudflare Python Worker for extracting P&ID piping tags with Gemini vision.

## Structure

- `src/main.py` - FastAPI Worker
- `pyproject.toml` - Python dependencies and Pywrangler
- `wrangler.jsonc` - Cloudflare Worker configuration

## Cloudflare setup

Create a new Worker connected to this repository.

Root directory: `/`

Build command:
```text
echo "No build step required"
```

Deploy command:
```text
uv run pywrangler deploy
```

Add Worker secret:
```text
GEMINI_API_KEY
```

Do not put the Gemini key in GitHub or frontend code.

## API

### GET /health

Returns Worker and model status.

### POST /api/extract

JSON body:
```json
{
  "overview": {
    "image": "<base64 or data URL>",
    "mime_type": "image/jpeg"
  },
  "tiles": [
    {
      "id": "tile-1",
      "image": "<base64 or data URL>",
      "mime_type": "image/jpeg"
    }
  ]
}
```

The response contains:
- `Tag No.`
- `P&ID No.`
- `From` (blank)
- `To` (blank)
- `NPS (in)`
- evidence/confidence for review

## Model

Primary: `gemini-3.8-flash`
Fallback: `gemini-3.7-flash`
