from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    redirect,
    session
)

import os
import logging

import requests

from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from google.cloud import firestore


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=16 * 1024,
)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

if not app.secret_key:
    raise RuntimeError(
        "SECRET_KEY is missing from your .env file"
    )


# ==================================================
# FIRESTORE
# ==================================================

# Uses the Google Application Default Credentials
# configured through gcloud auth application-default login
db = firestore.Client(project="bhiduai-3a24d")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MAX_QUESTION_LENGTH = 4_000


# ==================================================
# GOOGLE OAUTH
# ==================================================

oauth = OAuth(app)

google = oauth.register(
    name="google",
    server_metadata_url=(
        "https://accounts.google.com/.well-known/openid-configuration"
    ),
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid email profile"
    }
)


# ==================================================
# HTML INTERFACE
# ==================================================

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>BhiduAI — Your local AI companion</title>

    <style>
        * {
            box-sizing: border-box;
        }

        :root { --bg: #0b1020; --panel: rgba(18, 26, 51, .82); --surface: #171f3b; --text: #f4f7ff; --muted: #aab4ce; --line: rgba(183, 199, 255, .14); --accent: #8b7cff; --accent-2: #41d6c3; }
        body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at 82% 8%, #26346a 0, transparent 28rem), radial-gradient(circle at 5% 100%, #1a665f 0, transparent 30rem), var(--bg); color: var(--text); }

        .app {
            display: flex;
            min-height: 100vh;
        }

        .sidebar {
            width: 268px;
            background: rgba(9, 14, 31, .72);
            backdrop-filter: blur(18px);
            padding: 24px 16px;
            border-right: 1px solid var(--line);
            display: flex;
            flex-direction: column;
        }

        .logo {
            display: flex; align-items: center; gap: 10px; color: var(--text); font-size: 22px; font-weight: 750; letter-spacing: -.6px; margin: 0 8px 34px;
        }
        .logo-mark { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 11px; color: #10152b; background: linear-gradient(135deg, var(--accent), var(--accent-2)); font-size: 18px; }

        .new-chat {
            width: 100%;
            padding: 13px;
            border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.04); color: var(--text);
            cursor: pointer;
            font-size: 0;
        }
        .new-chat::after { content: "+  New chat"; font-size: 14px; }

        .new-chat:hover {
            background: rgba(139, 124, 255, .16); border-color: rgba(139, 124, 255, .55);
        }

        .account {
            margin-top: auto;
            padding-top: 20px;
            border-top: 1px solid var(--line);
        }

        .account-name {
            color: var(--text);
            font-size: 14px;
            margin-bottom: 5px;
            overflow-wrap: anywhere;
        }

        .account-email {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 12px;
            overflow-wrap: anywhere;
        }

        .login-button,
        .logout-button {
            display: block;
            width: 100%;
            padding: 11px;
            border-radius: 9px;
            text-align: center;
            text-decoration: none;
            cursor: pointer;
            font-size: 14px;
        }

        .login-button {
            background: linear-gradient(135deg, var(--accent), #a66bff); color: white; font-weight: 650;
        }

        .logout-button {
            border: 1px solid var(--line); color: var(--text); background: transparent;
        }

        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
            background: transparent;
        }

        .topbar {
            display: flex; align-items: center; justify-content: space-between; padding: 17px 34px; border-bottom: 1px solid var(--line); font-size: 15px; font-weight: 650; background: rgba(11,16,32,.45); backdrop-filter: blur(10px);
        }
        .online { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 12px; font-weight: 500; } .online i { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-2); box-shadow: 0 0 12px var(--accent-2); }

        .chat {
            flex: 1;
            overflow-y: auto;
            padding: 28px 34px;
        }

        .welcome {
            max-width: 790px;
            margin: 9vh auto;
            text-align: center;
        }

        .welcome h1 {
            font-size: clamp(32px, 5vw, 52px); letter-spacing: -1.8px; margin: 16px 0 12px;
        }

        .welcome p {
            color: var(--muted); font-size: 16px; margin: 8px;
        }
        .welcome-orb { display: grid; place-items: center; margin: auto; width: 70px; height: 70px; border-radius: 24px; font-size: 34px; background: linear-gradient(135deg, rgba(139,124,255,.9), rgba(65,214,195,.82)); box-shadow: 0 16px 60px rgba(77, 113, 255, .35); }
        .suggestions { display: flex; flex-wrap: wrap; justify-content: center; gap: 9px; margin: 30px auto 0; } .suggestion { border: 1px solid var(--line); border-radius: 100px; padding: 10px 14px; background: rgba(255,255,255,.045); color: #dce4ff; cursor: pointer; transition: .2s ease; } .suggestion:hover { transform: translateY(-2px); border-color: var(--accent); background: rgba(139,124,255,.17); }

        .message {
            max-width: 820px; margin: 0 auto 22px;
            line-height: 1.7;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }

        .user {
            background: linear-gradient(135deg, #6e5de0, #8775ef); padding: 14px 18px; border-radius: 18px 18px 4px 18px; margin-left: auto; width: fit-content; max-width: min(100%, 670px); box-shadow: 0 8px 25px rgba(63, 48, 160, .22);
        }

        .assistant {
            padding: 16px 18px; border: 1px solid var(--line); background: rgba(20, 29, 58, .65); border-radius: 18px 18px 18px 4px; box-shadow: 0 8px 25px rgba(0,0,0,.12);
        }

        .label {
            font-size: 13px;
            font-weight: bold;
            margin-bottom: 6px;
            color: var(--accent-2);
        }

        .input-area {
            padding: 18px 32px 25px; background: linear-gradient(transparent, rgba(11,16,32,.82) 18%);
        }

        .input-box {
            max-width: 820px;
            margin: auto;
            display: flex;
            gap: 10px;
            background: rgba(23,31,59,.92); border: 1px solid rgba(181,195,255,.23); border-radius: 18px; padding: 8px 9px 8px 16px; box-shadow: 0 12px 35px rgba(0,0,0,.2); transition: border-color .2s, box-shadow .2s;
        }
        .input-box:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(139,124,255,.16), 0 12px 35px rgba(0,0,0,.2); }

        textarea {
            flex: 1;
            resize: none;
            border: none;
            outline: none;
            font-family: inherit;
            font-size: 15px;
            padding: 10px;
            background: transparent;
            color: var(--text);
            min-height: 45px;
        }

        textarea::placeholder {
            color: #8e9ab8;
        }

        button.send {
            align-self: flex-end;
            border: none;
            border-radius: 13px; width: 44px; height: 44px; padding: 0; background: linear-gradient(135deg, var(--accent), #a66bff); color: white; font-size: 0;
            cursor: pointer;
            font-weight: bold;
        }

        button.send:hover {
            filter: brightness(1.12); transform: translateY(-1px);
        }

        button.send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .status {
            max-width: 780px;
            margin: 8px auto 0;
            color: var(--muted);
            font-size: 12px;
            text-align: center;
        }

        @media (max-width: 700px) {
            .sidebar {
                display: none;
            }

            .chat {
                padding: 18px 14px;
            }

            .input-area {
                padding: 12px;
            }

            .topbar {
                padding: 15px 18px;
            }
        }
    </style>
</head>

<body>

<div class="app">

    <aside class="sidebar">

        <div class="logo"><span class="logo-mark">&#10022;</span> BhiduAI</div>

        <button class="new-chat" onclick="newChat()">
            ＋ New chat
        </button>

        <div class="account">

            {% if user %}

                <div class="account-name">
                    {{ user.name or "Google user" }}
                </div>

                <div class="account-email">
                    {{ user.email }}
                </div>

                <a class="logout-button" href="/logout">
                    Sign out
                </a>

            {% else %}

                <a class="login-button" href="/login/google">
                    Sign in with Google
                </a>

            {% endif %}

        </div>

    </aside>


    <main class="main">

        <div class="topbar">
            <span>New conversation</span>
            <span class="online"><i></i> Local & private</span>
        </div>


        <div class="chat" id="chat">

            <div class="welcome" id="welcome">

                <div class="welcome-orb">&#10022;</div>
                <h1>What are we exploring today?</h1>

                <p>
                    Your local AI assistant powered by Ollama.
                </p>

                {% if user %}
                    <p>
                        Signed in as {{ user.email }}
                    </p>
                {% else %}
                    <p>
                        Sign in with Google to save your chat history.
                    </p>
                {% endif %}

                <div class="suggestions">
                    <button class="suggestion" onclick="usePrompt('Help me plan my day')">Plan my day</button>
                    <button class="suggestion" onclick="usePrompt('Explain a topic simply')">Explain a topic</button>
                    <button class="suggestion" onclick="usePrompt('Give me creative ideas')">Brainstorm ideas</button>
                </div>

            </div>

        </div>


        <div class="input-area">

            <div class="input-box">

                <textarea
                    id="question"
                    placeholder="Message BhiduAI..."
                    rows="1"
                    onkeydown="handleKey(event)"
                ></textarea>

                <button
                    class="send"
                    id="sendButton"
                    onclick="askAI()"
                >
                    &uarr;
                </button>

            </div>

            <div class="status">
                Chats are saved to your Google account.
            </div>

        </div>

    </main>

</div>


<script>

    function handleKey(event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            askAI();
        }
    }

    function usePrompt(prompt) {
        const input = document.getElementById("question");
        input.value = prompt;
        input.focus();
        resizeInput();
    }

    function resizeInput() {
        const input = document.getElementById("question");
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 150) + "px";
    }


    function addMessage(text, type) {
        const chat = document.getElementById("chat");

        const message = document.createElement("div");
        message.className = "message " + type;

        const label = document.createElement("div");
        label.className = "label";
        label.innerText = type === "user" ? "You" : "BhiduAI";

        const content = document.createElement("div");
        content.innerText = text;

        message.appendChild(label);
        message.appendChild(content);

        chat.appendChild(message);

        chat.scrollTop = chat.scrollHeight;

        return content;
    }


    async function loadHistory() {
        try {
            const response = await fetch("/history");

            if (!response.ok) {
                return;
            }

            const chats = await response.json();

            if (!Array.isArray(chats) || !chats.length) {
                return;
            }

            const welcome = document.getElementById("welcome");

            if (welcome) {
                welcome.remove();
            }

            for (const chat of chats) {
                addMessage(chat.question, "user");
                addMessage(chat.answer, "assistant");
            }

        } catch (error) {
            console.log("Could not load chat history:", error);
        }
    }


    async function askAI() {
        const input = document.getElementById("question");
        const button = document.getElementById("sendButton");

        const question = input.value.trim();

        if (!question) {
            return;
        }

        const welcome = document.getElementById("welcome");

        if (welcome) {
            welcome.remove();
        }

        input.value = "";
        input.disabled = true;
        button.disabled = true;

        addMessage(question, "user");

        const answerBox = addMessage("Thinking...", "assistant");

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question
                })
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok || data.error) {
                answerBox.innerText = "Error: " + (data.error || "Request failed.");
            } else {
                answerBox.innerText = data.answer;
            }

        } catch (error) {
            answerBox.innerText =
                "Could not connect to BhiduAI. Is Ollama running?";
        }

        finally {
            input.disabled = false;
            button.disabled = false;
            resizeInput();
            input.focus();
        }
    }


    function newChat() {
        document.getElementById("chat").innerHTML = `
            <div class="welcome" id="welcome">
                <div class="welcome-orb">&#10022;</div>
                <h1>What are we exploring today?</h1>
                <p>Your local AI assistant powered by Ollama.</p>
                <div class="suggestions">
                    <button class="suggestion" onclick="usePrompt('Help me plan my day')">Plan my day</button>
                    <button class="suggestion" onclick="usePrompt('Explain a topic simply')">Explain a topic</button>
                    <button class="suggestion" onclick="usePrompt('Give me creative ideas')">Brainstorm ideas</button>
                </div>
            </div>
        `;
    }


    window.addEventListener("DOMContentLoaded", () => {
        loadHistory();
        document.getElementById("question").addEventListener("input", resizeInput);
    });

</script>

</body>
</html>
"""


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template_string(
        HTML,
        user=session.get("user")
    )


# ==================================================
# GOOGLE LOGIN
# ==================================================

@app.route("/login/google")
def login_google():
    redirect_uri = request.url_root.rstrip("/") + "/auth/callback"

    return google.authorize_redirect(redirect_uri)


@app.route("/auth/callback")
def google_callback():
    try:
        token = google.authorize_access_token()

        user_info = token.get("userinfo")

        if not user_info:
            user_info = google.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                token=token
            ).json()

        # The "sub" value is Google's unique ID for the account.
        # It is safer than using the email as the database document ID.
        session["user"] = {
            "id": user_info.get("sub"),
            "name": user_info.get("name"),
            "email": user_info.get("email"),
            "picture": user_info.get("picture")
        }

        return redirect("/")

    except Exception:
        logger.exception("Google login failed")
        return "Google login failed. Please try again.", 500


# ==================================================
# LOGOUT
# ==================================================

@app.route("/logout")
def logout():
    session.clear()

    return redirect("/")


# ==================================================
# LOAD CHAT HISTORY
# ==================================================

@app.route("/history")
def history():
    user = session.get("user")

    if not user or not user.get("id"):
        return jsonify([])

    try:
        chats = (
            db.collection("users")
            .document(user["id"])
            .collection("chats")
            .order_by(
                "created_at",
                direction=firestore.Query.DESCENDING
            )
            .limit(50)
            .stream()
        )

        result = []

        for chat in chats:
            data = chat.to_dict()

            result.append({
                "question": data.get("question", ""),
                "answer": data.get("answer", "")
            })

        # Show oldest of the last 50 messages first.
        result.reverse()

        return jsonify(result)

    except Exception:
        logger.exception("Could not load chat history")

        return jsonify({
            "error": "Could not load chat history."
        }), 500


# ==================================================
# ASK OLLAMA AND SAVE CHAT
# ==================================================

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user = session.get("user")

        if not user or not user.get("id"):
            return jsonify({
                "error": "Please sign in with Google first."
            }), 401

        data = request.get_json(silent=True) or {}

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "error": "Please enter a message."
            }), 400

        if len(question) > MAX_QUESTION_LENGTH:
            return jsonify({"error": "Please keep messages under 4,000 characters."}), 400

        # Send the question to the local Ollama model.
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": question,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get(
            "response",
            "No response received."
        )

        
        db.collection("users") \
            .document(user["id"]) \
            .collection("chats") \
            .add({
                "question": question,
                "answer": answer,
                "created_at": firestore.SERVER_TIMESTAMP
            })

        return jsonify({
            "answer": answer
        })

    except requests.exceptions.RequestException:
        return jsonify({
            "error": (
                "Ollama is not running. "
                "Start Ollama and try again."
            )
        }), 500

    except Exception:
        logger.exception("Could not answer question")

        return jsonify({
            "error": "Something went wrong while processing your message."
        }), 500


# ==================================================
# START FLASK
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
