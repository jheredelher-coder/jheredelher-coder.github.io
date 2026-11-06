import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Solo permite solicitudes desde tu sitio y desde localhost para pruebas.
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://jheredelher-coder.github.io,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.7")

SYSTEM_PROMPT = (
    "Eres JHEREMYAS AI, un asistente útil, claro y respetuoso. "
    "Responde en español salvo que el usuario pida otro idioma. "
    "Explica paso a paso cuando sea útil y reconoce cuando no sabes algo."
)

def clean_history(raw_history):
    if not isinstance(raw_history, list):
        raise ValueError("El historial debe ser una lista.")

    cleaned = []
    for item in raw_history[-20:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if len(content) > 4000:
            raise ValueError("Cada mensaje puede tener como máximo 4000 caracteres.")
        cleaned.append({"role": role, "content": content})

    if not cleaned or cleaned[-1]["role"] != "user":
        raise ValueError("No se encontró el último mensaje del usuario.")
    return cleaned

def ask_gemini(history):
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta configurar GEMINI_API_KEY en las variables privadas del servidor.")

    contents = []
    for item in history:
        contents.append({
            "role": "user" if item["role"] == "user" else "model",
            "parts": [{"text": item["content"]}]
        })

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + GEMINI_MODEL
        + ":generateContent"
    )
    response = requests.post(
        url,
        params={"key": GEMINI_API_KEY},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1200}
        },
        timeout=90
    )

    if not response.ok:
        raise RuntimeError("Gemini API: " + str(response.status_code) + " - " + response.text[:500])

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini no devolvió una respuesta. Revisa el acceso al modelo o sus límites.")

    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not answer.strip():
        raise RuntimeError("Gemini devolvió una respuesta vacía.")
    return answer.strip()

def ask_grok(history):
    if not XAI_API_KEY:
        raise RuntimeError("Falta configurar XAI_API_KEY en las variables privadas del servidor.")

    response = requests.post(
        "https://api.x.ai/v1/responses",
        headers={
            "Authorization": "Bearer " + XAI_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "model": GROK_MODEL,
            "instructions": SYSTEM_PROMPT,
            "input": [
                {"role": item["role"], "content": item["content"]}
                for item in history
            ],
            "store": False
        },
        timeout=120
    )

    if not response.ok:
        raise RuntimeError("Grok API: " + str(response.status_code) + " - " + response.text[:500])

    data = response.json()
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    text_parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    text_parts.append(part["text"])
    answer = "\n".join(text_parts).strip()
    if not answer:
        raise RuntimeError("Grok no devolvió texto. Revisa la respuesta de la API.")
    return answer

@app.get("/")
def home():
    return jsonify({"service": "JHEREMYAS AI", "status": "online"})

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        provider = data.get("provider")
        if provider not in ("gemini", "grok"):
            return jsonify({"error": "Modelo no válido. Elige Gemini o Grok."}), 400

        history = clean_history(data.get("history"))
        answer = ask_gemini(history) if provider == "gemini" else ask_grok(history)
        return jsonify({"reply": answer, "provider": provider})

    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except requests.Timeout:
        return jsonify({"error": "La API tardó demasiado en responder. Inténtalo otra vez."}), 504
    except requests.RequestException:
        return jsonify({"error": "No se pudo contactar con el proveedor de IA. Inténtalo más tarde."}), 502
    except RuntimeError as error:
        # No se devuelven claves ni variables privadas al navegador.
        return jsonify({"error": str(error)}), 502
    except Exception:
        app.logger.exception("Error inesperado en /api/chat")
        return jsonify({"error": "Ocurrió un error interno en el servidor."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
