"""LLaVA vision-language model client via Ollama.

Sends chest X-ray images to a local LLaVA instance for visual description
and finding extraction using multimodal prompting.
"""

import base64
import io
import os

import httpx
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_VISION = os.getenv("MODEL_VISION", "llava")


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.thumbnail((512, 512), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def describe_xray(image: Image.Image) -> str:
    """Send image to LLaVA and get clinical description."""
    b64_image = _image_to_base64(image)

    payload = {
        "model": MODEL_VISION,
        "prompt": (
            "This is a frontal (PA or AP) chest X-ray. Describe up to 5 visible "
            "findings as short bullet points. Do not give a diagnosis. "
            "If unsure, say 'uncertain'."
        ),
        "images": [b64_image],
        "stream": False,
        "keep_alive": 300,
    }

    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            description = result.get("response", "").strip()

            print(f"LLaVA response length: {len(description)}")
            print(f"LLaVA preview: {description[:150]}")

            if len(description) < 10:
                return (
                    "LLaVA description unavailable — "
                    "using vision model scores only."
                )

            client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": MODEL_VISION, "keep_alive": 0},
            )

            return description

    except httpx.ConnectError:
        raise RuntimeError(
            "Ollama not running. Start with: ollama serve"
        )
    except httpx.TimeoutException:
        return (
            "LLaVA timed out — using vision model "
            "scores only."
        )
    except Exception as e:
        print(f"LLaVA error: {e}")
        return (
            "LLaVA description unavailable — "
            "using vision model scores only."
        )
