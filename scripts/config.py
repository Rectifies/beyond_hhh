"""
Configuration for Beyond HHH experiments.
API keys are loaded from ../.env
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model used for model-as-judge safety evaluation
SAFETY_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Models tested
MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "model_id": "chatgpt-4o-latest",
    },
    "gpt-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2-2025-12-11",
    },
    "claude-opus-4.6": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
    },
    "gemini-3-pro-preview": {
        "provider": "google",
        "model_id": "gemini-3-pro-preview",
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
    },
}

# Paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DATA_DIR = f"{PROJECT_ROOT}/data"
