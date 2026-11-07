"""Model backends.

Each backend hides how one model provider's API works. The agent loop in
agent.py only needs three things from a backend:

    start(question)                              -> a new conversation
    next_turn(conversation)                      -> the model's reply (Turn)
    add_results(conversation, calls, results)    -> send tool results back

To add another provider, write a class with these three methods and a `label`.
"""
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional


class BackendError(Exception):
    """Something went wrong talking to the model. The message says what to do."""


class QuotaExhausted(BackendError):
    """The API refused the request because of a rate or daily limit."""


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Turn:
    text: Optional[str]      # the model's text, if any
    calls: List[ToolCall]    # tools the model wants run (empty means final answer)


# ---------------------------------------------------------------------------
# Gemini (Google's hosted API)
# ---------------------------------------------------------------------------
class GeminiBackend:
    def __init__(self, system_prompt, tool_declarations):
        from google import genai
        from google.genai import errors, types

        self._genai, self._errors, self._types = genai, errors, types
        self.model = os.environ.get("SQL_AGENT_MODEL", "gemini-2.5-flash")
        self.label = self.model
        # Minimum seconds between calls. On a free tier, set this so you stay
        # under the requests-per-minute limit (for example 7 for 10 per minute).
        self.min_interval = float(os.environ.get("SQL_AGENT_MIN_INTERVAL", "0"))
        self._last_call = 0.0
        self._client = None
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=tool_declarations)],
            # We run the tool loop ourselves so we can log and count every step.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0,
        )

    def _get_client(self):
        if self._client is None:
            # Function calling needs the v1beta API. Some SDK versions or settings
            # default to v1, which rejects tool calls, so we set it explicitly.
            self._client = self._genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
                http_options=self._types.HttpOptions(api_version="v1beta"),
            )
        return self._client

    def start(self, question):
        types = self._types
        return [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    def next_turn(self, conversation):
        response = self._generate(conversation)
        calls = response.function_calls or []
        if calls:
            # Keep the model's tool request in the history so it can be answered.
            conversation.append(response.candidates[0].content)
        return Turn(
            text=response.text if not calls else None,
            calls=[ToolCall(c.name, dict(c.args or {})) for c in calls],
        )

    def add_results(self, conversation, calls, results):
        types = self._types
        parts = []
        for call, result in zip(calls, results):
            payload = result if isinstance(result, dict) else {"result": result}
            parts.append(types.Part.from_function_response(name=call.name, response=payload))
        conversation.append(types.Content(role="user", parts=parts))

    def _generate(self, conversation):
        """Call the model.

        - Per-minute rate limit (429): wait and try again.
        - Daily limit (429 mentioning PerDay): stop at once with QuotaExhausted.
        - Temporary server errors (5xx): wait and try again.
        - Anything else (bad request, bad key): raise immediately.
        """
        errors = self._errors
        for attempt in range(5):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                return self._get_client().models.generate_content(
                    model=self.model, contents=conversation, config=self._config
                )
            except errors.ClientError as e:
                if e.code != 429:
                    raise
                if "PerDay" in str(e):
                    raise QuotaExhausted(
                        "Daily request limit reached. It resets at midnight Pacific time."
                    ) from e
                time.sleep(15 * (attempt + 1))
            except errors.ServerError:
                time.sleep(5 * (attempt + 1))
        raise QuotaExhausted("Still rate limited after several waits. Try again in a few minutes.")


# ---------------------------------------------------------------------------
# Ollama (open-source models running on your own computer)
# ---------------------------------------------------------------------------
class OllamaBackend:
    def __init__(self, system_prompt, tool_declarations):
        self.model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
        self.label = f"ollama:{self.model}"

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        self.host = host.replace("0.0.0.0", "localhost")

        # Ollama's default context window is small and silently cuts off long
        # conversations. The schema plus the chat history needs more room.
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
        # Seconds to wait for one reply. The first call is slow while the model loads.
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
        # Only for models that can "think" (such as qwen3). Leave unset to use the
        # model's default. Set OLLAMA_THINK=false to make it faster.
        think = os.environ.get("OLLAMA_THINK", "").strip().lower()
        self.think = {"true": True, "false": False}.get(think)

        self.system_prompt = system_prompt
        self.tools = [{"type": "function", "function": d} for d in tool_declarations]

    def start(self, question):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

    def next_turn(self, conversation):
        payload = {
            "model": self.model,
            "messages": conversation,
            "tools": self.tools,
            "stream": False,
            # temperature 0 and a fixed seed keep runs as repeatable as possible
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            payload["think"] = self.think

        reply = self._post("/api/chat", payload)
        message = reply.get("message") or {}
        conversation.append(message)  # keep the model's reply in the history

        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some servers send arguments as a JSON string
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(fn.get("name", ""), args))

        # Some models put their reasoning inline in <think> tags. Remove it.
        text = re.sub(r"<think>.*?</think>", "", message.get("content") or "", flags=re.S).strip()
        return Turn(text=text or None, calls=calls)

    def add_results(self, conversation, calls, results):
        for call, result in zip(calls, results):
            content = result if isinstance(result, str) else json.dumps(result, default=str)
            conversation.append({"role": "tool", "tool_name": call.name, "content": content})

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                message = json.loads(body).get("error", body)
            except ValueError:
                message = body
            raise BackendError(self._explain(e.code, str(message))) from e
        except urllib.error.URLError as e:
            raise BackendError(
                f"Could not reach Ollama at {self.host}. Start the Ollama app "
                "(or run 'ollama serve') and try again."
            ) from e
        except (socket.timeout, TimeoutError) as e:
            raise BackendError(
                f"Ollama did not answer within {self.timeout:.0f} seconds. The model may be too "
                "large for your GPU. Try a smaller model, or raise OLLAMA_TIMEOUT."
            ) from e

    def _explain(self, code, message):
        low = message.lower()
        if code == 404 or "not found" in low:
            return f"Ollama does not have the model '{self.model}'. Download it with:  ollama pull {self.model}"
        if "does not support tools" in low:
            return (f"The model '{self.model}' does not support tool calling, so it cannot run "
                    "this agent. Choose a model that lists tool support in the Ollama library.")
        if "does not support thinking" in low:
            return f"The model '{self.model}' does not support thinking. Remove OLLAMA_THINK from .env."
        return f"Ollama returned an error ({code}): {message}"


# ---------------------------------------------------------------------------
def get_backend(system_prompt, tool_declarations):
    """Pick the backend named by SQL_AGENT_BACKEND in .env (default: gemini)."""
    name = os.environ.get("SQL_AGENT_BACKEND", "gemini").strip().lower()
    if name == "gemini":
        return GeminiBackend(system_prompt, tool_declarations)
    if name == "ollama":
        return OllamaBackend(system_prompt, tool_declarations)
    raise BackendError(f"Unknown SQL_AGENT_BACKEND '{name}'. Use 'gemini' or 'ollama'.")
