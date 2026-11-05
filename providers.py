"""AI provider abstraction. Keys are read from server environment variables only.
Add a provider = subclass Provider. No provider is required: offline mode always works."""
import os, json, urllib.request, urllib.error

class ProviderError(Exception):
    pass

class Provider:
    name = "base"
    def available(self): return False
    def complete(self, system: str, prompt: str, json_mode: bool = False) -> str: raise NotImplementedError

class OfflineProvider(Provider):
    name = "offline"
    def available(self): return True
    def complete(self, *a, **k): raise ProviderError("Offline mode has no language model.")

def _post(url, payload, headers, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ProviderError(f"Provider returned HTTP {e.code} (check key, model name, or free-tier rate limit).")
    except Exception as e:
        raise ProviderError(f"Provider unreachable: {e}")

class OpenAICompatible(Provider):
    """Works with Groq, Google Gemini (OpenAI-compatible endpoint), OpenRouter, and local servers."""
    def __init__(self, name, base_url, key_env, model_env, default_model, needs_key=True):
        self.name, self.base_url, self.key_env = name, base_url.rstrip("/"), key_env
        self.model = os.environ.get(model_env, default_model); self.needs_key = needs_key
    def _key(self): return os.environ.get(self.key_env, "")
    def available(self): return bool(self._key()) or not self.needs_key
    def complete(self, system, prompt, json_mode=False):
        payload = {"model": self.model, "temperature": 0.2,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode: payload["response_format"] = {"type": "json_object"}
        h = {"Authorization": f"Bearer {self._key()}"} if self._key() else {}
        data = _post(f"{self.base_url}/chat/completions", payload, h)
        try: return data["choices"][0]["message"]["content"]
        except Exception: raise ProviderError("Unexpected response from provider.")

def registry():
    return {
        "offline": OfflineProvider(),
        "groq": OpenAICompatible("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile"),
        "gemini": OpenAICompatible("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.0-flash"),
        "openrouter": OpenAICompatible("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        "ollama": OpenAICompatible("ollama", os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"), "OLLAMA_API_KEY", "OLLAMA_MODEL", "llama3.2", needs_key=False),
    }

def get_provider(name=None):
    name = name or os.environ.get("NEURO_AI_PROVIDER", "offline")
    p = registry().get(name)
    return p if p is not None else OfflineProvider()

def status():
    return [{"name": n, "configured": p.available(), "model": getattr(p, "model", None)} for n, p in registry().items()]
