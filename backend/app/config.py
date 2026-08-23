import os
from pathlib import Path

from dotenv import load_dotenv


# -----------------------------
# BASE DIRECTORY
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from backend/.env
load_dotenv(BASE_DIR / ".env")


# -----------------------------
# DATABASE
# -----------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'priorauth.db'}",
)


# -----------------------------
# JWT
# -----------------------------

JWT_SECRET = os.getenv(
    "JWT_SECRET",
    "change-me-in-production-please",
)

JWT_ALGORITHM = "HS256"

ACCESS_TOKEN_MINUTES = int(
    os.getenv("ACCESS_TOKEN_MINUTES", "720")
)


# -----------------------------
# FILE UPLOADS
# -----------------------------

UPLOAD_DIR = Path(
    os.getenv(
        "UPLOAD_DIR",
        str(BASE_DIR / "uploads"),
    )
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------
# ML MODELS
# -----------------------------

MODELS_DIR = BASE_DIR / "ml" / "models"


# -----------------------------
# CORS
# -----------------------------

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")


# -----------------------------
# MEDICAL NECESSITY
# -----------------------------

AUTO_APPROVE_MIN_POLICY_FIT = float(
    os.getenv(
        "AUTO_APPROVE_MIN_POLICY_FIT",
        "0.62",
    )
)

AUTO_DENY_MAX_POLICY_FIT = float(
    os.getenv(
        "AUTO_DENY_MAX_POLICY_FIT",
        "0.38",
    )
)

MIN_DOCUMENTATION_SCORE = float(
    os.getenv(
        "MIN_DOCUMENTATION_SCORE",
        "0.75",
    )
)


# -----------------------------
# GROQ / AI
# -----------------------------

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# -----------------------------
# MODEL 1 -- guideline reasoning service (Render)
# -----------------------------

PRIOR_AUTH_URL = os.getenv(
    "PRIOR_AUTH_URL",
    "https://prior-auth-api-bmju.onrender.com",
).rstrip("/")

PRIOR_AUTH_TOKEN = os.getenv(
    "PRIOR_AUTH_TOKEN",
    "",
).strip()

# Measured against the live free-tier instance: cold start 50-90s, warm
# /analyze ~54s. 120s leaves headroom without hanging a worker indefinitely.
PRIOR_AUTH_READ_TIMEOUT = float(
    os.getenv("PRIOR_AUTH_READ_TIMEOUT", "120")
)

PRIOR_AUTH_CONNECT_TIMEOUT = float(
    os.getenv("PRIOR_AUTH_CONNECT_TIMEOUT", "10")
)

# Wake the instance at startup so the first real request does not pay the
# cold start. Turn off for local development against a warm instance.
PRIOR_AUTH_WARM_ON_STARTUP = os.getenv(
    "PRIOR_AUTH_WARM_ON_STARTUP", "1"
) not in ("0", "false", "False")

# Append the guideline rules to the decision ledger as extra rows. The necessity
# score is computed before they are added, so this changes what a reviewer sees
# and not what the engine decides. Set to 0 if the ledger gets too long.
INCLUDE_GUIDELINE_CRITERIA = os.getenv(
    "INCLUDE_GUIDELINE_CRITERIA", "1"
) not in ("0", "false", "False")


# -----------------------------
# MODEL 2 -- appeal-propensity regressor (HistGradientBoostingRegressor)
# -----------------------------

MODEL2_PATH = Path(
    os.getenv(
        "MODEL2_PATH",
        str(BASE_DIR / "ml" / "models" / "appeal_propensity.joblib"),
    )
)

MODEL2_METRICS_PATH = Path(
    os.getenv(
        "MODEL2_METRICS_PATH",
        str(BASE_DIR / "ml" / "models" / "appeal_metrics.json"),
    )
)

MODEL2_ENABLED = os.getenv(
    "MODEL2_ENABLED", "1"
) not in ("0", "false", "False")

# A denial whose reappeal risk lands above this percentile of the training
# population goes to a human even when nothing is fixable by documentation.
MODEL2_REAPPEAL_PERCENTILE = float(
    os.getenv("MODEL2_REAPPEAL_PERCENTILE", "80")
)
