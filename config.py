import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LiveKit Configuration
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # Agent dispatch. Empty registers the worker for automatic dispatch to any
    # room. A non-empty value enables explicit dispatch: the worker only accepts
    # jobs for rooms requesting this name. Must match frontend AGENT_NAME.
    agent_name: str = ""

    # LLM Configuration
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # Document Ingestion
    docs_dir: str = "docs"

    # Embedding & Vector Store
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # The RAG index is built into the Docker image and used locally at runtime.
    # Use an absolute path so build-time and LiveKit runtime resolve the same
    # location regardless of the process working directory.
    qdrant_path: str = "/app/data/qdrant"

    # Kept for compatibility with existing environment configuration.
    # The packaged LiveKit deployment intentionally uses the local Qdrant index.
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    collection_name: str = "telecom_docs"
    chunk_size: int = 500

    # Voice Configuration
    stt_model: str = "deepgram/nova-3"
    stt_language: str = "multi"
    tts_model: str = "inworld/inworld-tts-2"
    tts_voice: str = "Ashley"

    # Browser automation (Playwright / Chromium)
    # All browser_* tools are gated by `browser_enabled`. When False the
    # browser_* LangChain tools are not registered and the SYSTEM_PROMPT
    # omits the BROWSER TOOLS section, so the agent remains fully usable
    # in environments without Chromium installed.
    browser_enabled: bool = True
    browser_headless: bool = True
    browser_max_contexts: int = 8
    browser_idle_timeout_s: int = 300
    browser_allowed_domains: str = ""  # CSV globs; empty = allow all + warn
    browser_allow_private: bool = False
    browser_allow_eval: bool = True
    browser_default_timeout_ms: int = 15000
    browser_action_timeout_ms: int = 5000
    browser_viewport_w: int = 1280
    browser_viewport_h: int = 720
    browser_locale: str = "en-IN"
    browser_timezone: str = "Asia/Kolkata"
    browser_user_agent: str = ""  # empty = Playwright default
    browser_proxy: str = ""
    browser_download_dir: str = "/tmp/browser_downloads"
    browser_max_actions_per_turn: int = 25


settings = Settings()

# Set environment variables for LiveKit
os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
os.environ.setdefault("AGENT_NAME", settings.agent_name)
os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
