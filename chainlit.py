import httpx
import logging

logger = logging.getLogger(__name__)

# Replace with actual Ollama API URL (e.g., "http://localhost:8000" or another endpoint)
url = "http://localhost:8000"

def test_ollama_connection():
    try:
        print(f"Testing connection to Ollama API at {url}...")
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
        print("Successfully connected to Ollama service.")
        print(f"Response: {response.text}")
    except httpx.RequestError as exc:
        logger.error(f"Request error: {exc}")
    except httpx.HTTPStatusError as exc:
        logger.error(f"HTTP error: {exc.response.status_code} - {exc.response.text}")

test_ollama_connection()
