from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, PRIOR_AUTH_URL, PRIOR_AUTH_WARM_ON_STARTUP
from .database import Base, engine
from .routers import auth, chat, dashboard, requests, review, validation
from .services import ml, prior_auth_client
from .routers.admin import router as admin_router

app = FastAPI(
    title="Prior Authorization Intelligence Platform",
    description=(
        "AI-assisted prior authorization automation with PDF extraction, "
        "medical necessity evaluation, ML scoring, reviewer routing, "
        "appeal prediction, contextual validation and audit trails."
    ),
    version="2.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in CORS_ORIGINS
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth.router)
app.include_router(requests.router)
app.include_router(review.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(validation.router)
app.include_router(admin_router)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)

    if PRIOR_AUTH_WARM_ON_STARTUP:
        # Wake the guideline service now rather than making the first real user
        # wait out a cold start. Runs on a background thread; a failed warm-up
        # must not block start-up.
        prior_auth_client.warm()

    ready = ml.models_ready()

    print("\n==========================================")
    print(" PRIOR AUTHORIZATION PLATFORM")
    print("==========================================")
    print(
        "Model 1  guideline service:",
        "REACHABLE" if ready["policy_fit"] else "UNREACHABLE",
    )
    print("         ", PRIOR_AUTH_URL)
    print(
        "Model 2  supporting-material:",
        "READY" if ready["appeal_propensity"] else "MISSING",
    )

    if not ready["policy_fit"]:
        print(
            "\nWARNING: the guideline service is not answering. It may be waking "
            "from idle; adjudication will return 503 until it does."
        )
    if not ready["appeal_propensity"]:
        print(
            "\nWARNING: Model 2 did not load. Denials will stand as auto-denied "
            "without a supporting-material check."
        )

    print("==========================================\n")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "prior-authorization-platform",
        "models": ml.models_ready(),
    }
