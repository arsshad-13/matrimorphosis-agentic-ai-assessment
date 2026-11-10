
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from pydantic import BaseModel

from app.telemetry import (
    RUN_STEPS,
    HTTP_DURATION,
    HTTP_REQUESTS,
    configure_logging,
    log,
    run_config,
    setup_telemetry,
    shutdown_telemetry,
)

setup_telemetry()
configure_logging()
HTTPXClientInstrumentor().instrument()

from app.agent import build_agent, charts_from, clean_answer, steps_from
from app.debug_routes import install_routes
from app.context import Ctx
from app.errors import DocError, http_status
from app.kb import files
from app.kb.store import KnowledgeBase
from app.persistence import open_checkpointer
from app.schemas import AgentRequest, AgentResponse, KBResponse, UploadResponse
from app.tools_s1 import CHARTS_DIR
from app.users import InvalidCredentials, UserStore, UsernameTaken


# ---------------------------------------------------------
# Request schemas for user registration and login
# ---------------------------------------------------------

class UserCreateRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.kb = KnowledgeBase()
    app.state.users = UserStore()

    async with open_checkpointer() as saver:
        agent, tools = await build_agent(
            app.state.kb,
            checkpointer=saver,
        )

        app.state.agent, app.state.tools = agent, tools
        app.state.tool_names = [t.name for t in tools]

        log.info("startup_complete", tools=app.state.tool_names)

        yield

    shutdown_telemetry()


app = FastAPI(
    title="Agentic AI assessment - Document Agent (tools + knowledge base + MCP)",
    lifespan=lifespan,
)

FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
install_routes(app)


# ---------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------

async def get_current_user(request: Request) -> dict:
    """Validate a bearer token and return its authenticated user."""

    authorization = request.headers.get("Authorization", "")
    parts = authorization.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    token = parts[1]

    user = await asyncio.to_thread(
        request.app.state.users.user_for_token,
        token,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )

    request.state.auth_token = token
    return user


# ---------------------------------------------------------
# Thread ownership helper - Task 6
# ---------------------------------------------------------

async def claim_thread_or_forbid(
    request: Request,
    thread_id: str,
    user_id: str,
) -> None:
    """Claim a thread for its first user and reject access by other users."""

    claimed = await asyncio.to_thread(
        request.app.state.users.claim_thread,
        thread_id,
        user_id,
    )

    if not claimed:
        raise HTTPException(
            status_code=403,
            detail="This thread belongs to another user",
        )


# ---------------------------------------------------------
# User registration and authentication endpoints
# ---------------------------------------------------------

@app.post("/users", status_code=201)
async def create_user(req: UserCreateRequest, request: Request):
    """Register a new user."""

    try:
        result = await asyncio.to_thread(
            request.app.state.users.create,
            req.username,
            req.password,
        )
        return result

    except UsernameTaken as exc:
        raise HTTPException(
            status_code=409,
            detail="Username already taken",
        ) from exc

    except InvalidCredentials as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post("/login")
async def login(req: LoginRequest, request: Request):
    """Authenticate a user and issue an access token."""

    result = await asyncio.to_thread(
        request.app.state.users.login,
        req.username,
        req.password,
    )

    if result is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    return result


@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's details."""

    return {
        "user_id": user["user_id"],
        "username": user["username"],
    }


@app.post("/logout", status_code=204)
async def logout(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Revoke only the current authentication token."""

    token = request.state.auth_token

    await asyncio.to_thread(
        request.app.state.users.revoke,
        token,
    )

    return Response(status_code=204)


@app.delete("/users/me", status_code=204)
async def delete_my_account(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Delete the authenticated user's account and associated data."""

    user_id = user["user_id"]

    # Remove the user's knowledge base.
    await asyncio.to_thread(
        request.app.state.kb.reset,
        user_id,
    )

    # Remove the user, their tokens, and thread claims.
    await asyncio.to_thread(
        request.app.state.users.delete_user,
        user_id,
    )

    return Response(status_code=204)


# ---------------------------------------------------------
# HTTP request observability middleware
# ---------------------------------------------------------

@app.middleware("http")
async def observe(request: Request, call_next):
    """Per-request: request_id in every log line, API-hit metrics by route/method/status."""

    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=rid)

    t0, status = time.perf_counter(), 500

    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response

    finally:
        route = getattr(
            request.scope.get("route"),
            "path",
            request.url.path,
        )

        elapsed = time.perf_counter() - t0

        attrs = {
            "http.route": route,
            "http.method": request.method,
            "http.status_code": str(status),
        }

        if route != "/health":
            HTTP_REQUESTS.add(1, attrs)
            HTTP_DURATION.record(elapsed, attrs)

            log.info(
                "http_request",
                route=route,
                method=request.method,
                status=status,
                duration_ms=round(elapsed * 1000),
            )


# ---------------------------------------------------------
# Agent endpoint
# ---------------------------------------------------------

@app.post("/agent", response_model=AgentResponse)
async def agent(
    req: AgentRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Run the agent for an authenticated user with thread ownership."""

    user_id = user["user_id"]

    # Ensure this thread belongs to the authenticated user.
    await claim_thread_or_forbid(
        request,
        req.thread_id,
        user_id,
    )

    structlog.contextvars.bind_contextvars(thread_id=req.thread_id)

    result = await _guard(
        request.app.state.agent.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=run_config(thread_id=req.thread_id),
            context=Ctx(
                session_id=req.thread_id,
                user_id=user_id,
                token=request.state.auth_token,
            ),
        )
    )

    msgs = result["messages"]

    turn = msgs[
        max(i for i, m in enumerate(msgs) if m.type == "human"):
    ]

    steps = steps_from(turn)

    RUN_STEPS.record(len(steps))

    log.info(
        "agent_done",
        steps=len(steps),
        tools=[s["tool"] for s in steps],
        answer_chars=len(str(msgs[-1].content)),
    )

    return AgentResponse(
        answer=clean_answer(str(msgs[-1].content)),
        steps=steps,
        charts=charts_from(turn),
    )


# ---------------------------------------------------------
# Document upload endpoint
# ---------------------------------------------------------

@app.post("/upload", response_model=UploadResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user: dict = Depends(get_current_user),
):
    """Store an uploaded file for a thread owned by the authenticated user."""

    user_id = user["user_id"]

    # Claim or verify ownership before processing the upload.
    await claim_thread_or_forbid(
        request,
        session_id,
        user_id,
    )

    data = await file.read(files.MAX_BYTES + 1)

    try:
        file_id, path = files.save_upload(
            session_id,
            file.filename or "file",
            data,
        )

    except DocError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{exc.message} {exc.next_step}",
        ) from exc

    log.info(
        "file_uploaded",
        filename=path.name.split("__", 1)[1],
        size=len(data),
    )

    return UploadResponse(
        file_id=file_id,
        filename=path.name.split("__", 1)[1],
        size=len(data),
    )


# ---------------------------------------------------------
# Knowledge-base endpoints - Task 6: per-user KB
# ---------------------------------------------------------

@app.get("/kb/{session_id}", response_model=KBResponse)
async def kb_state(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Return the authenticated user's knowledge base."""

    # The KB is owned by user_id, not by the URL session_id.
    user_id = user["user_id"]
    kb: KnowledgeBase = request.app.state.kb

    prof = await asyncio.to_thread(
        kb.get_profile,
        user_id,
    ) or {
        "overview": None,
        "keywords": [],
    }

    documents = await asyncio.to_thread(
        kb.documents,
        user_id,
    )

    prompt_block = await asyncio.to_thread(
        kb.prompt_block,
        user_id,
    )

    return KBResponse(
        **prof,
        documents=documents,
        prompt_block=prompt_block,
    )


@app.delete("/kb/{session_id}")
async def kb_reset(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Clear only the authenticated user's knowledge base."""

    user_id = user["user_id"]

    await asyncio.to_thread(
        request.app.state.kb.reset,
        user_id,
    )

    return {"status": "cleared"}


# ---------------------------------------------------------
# Shared error handling
# ---------------------------------------------------------

async def _guard(awaitable):
    try:
        return await awaitable

    except Exception as exc:
        log.error(
            "request_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
        )

        status, message, headers = http_status(exc)

        raise HTTPException(
            status_code=status,
            detail=message,
            headers=headers,
        ) from exc
