import os
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_INDEX = ROOT_DIR / "index.html"
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq deprecated llama-3.1-70b-versatile; switch to the recommended successor.
DEFAULT_MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = """
You are an AI learning mentor specializing in Deep Learning, Computer Vision,
Reinforcement Learning, Transformers, and Generative AI.

Always respond in this format with numbered sections:

1. Concept Explanation
   - Clear overview with examples or analogies
   - Mention real-world use cases

2. Visual/Diagram Suggestion
   - Suggest diagrams or simple ASCII sketches

3. Step-by-Step Code
   - Ready-to-run Python with comments
   - Explain key steps before/after the code block

4. Mini Project/Application
   - Describe a small project/exercise
   - Explain how to test/validate results

5. Practical Tips
   - Bullet common pitfalls, best practices, datasets, hyperparameters, libraries

6. Interactive Guidance
   - Offer to answer questions or propose alternatives

Keep the tone beginner-friendly, simple, and practical.
""".strip()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route("/", methods=["GET"])
    def root():
        """Serve the frontend dashboard if it exists, otherwise fallback JSON."""
        if FRONTEND_INDEX.exists():
            return send_file(FRONTEND_INDEX)

        return jsonify(
            {
                "message": "Smart AI Mentor backend is running.",
                "endpoints": {
                    "health": "/health",
                    "mentor": "/api/mentor",
                },
            }
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/mentor", methods=["POST"])
    def mentor():
        payload = request.get_json(silent=True) or {}
        topic = (payload.get("topic") or "").strip()

        if not topic:
            return jsonify({"error": "Missing topic."}), 400

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return jsonify({"error": "GROQ_API_KEY is not set."}), 500

        groq_payload = {
            "model": DEFAULT_MODEL,
            "temperature": 0.4,
            "max_tokens": 1200,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nCreate a detailed guide following the required numbered format.",
                },
            ],
        }

        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                timeout=60,
                json=groq_payload,
            )
        except requests.RequestException as error:
            return jsonify({"error": f"Unable to reach Groq API: {error}"}), 502

        if not response.ok:
            return jsonify(
                {
                    "error": "Groq API returned an error.",
                    "details": response.text,
                }
            ), response.status_code

        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            return jsonify({"error": "No content returned from Groq API."}), 502

        return jsonify({"content": content})

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

