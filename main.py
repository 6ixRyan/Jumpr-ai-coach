"""
JUMPR AI Coach - FastAPI Backend
Powered by Claude (Anthropic)

This is the AI Coach endpoint for the JUMPR basketball training app.
It receives user messages and returns basketball coaching responses.

Cost/abuse controls (see the constants below): every reply is capped to
MAX_OUTPUT_TOKENS, every request's prompt text is bounded, and requests are
rate-limited per client. NOTE: these bound per-request and per-minute cost but
are NOT a hard dollar ceiling — set a monthly spend limit on the Anthropic
Console for the key this service uses. This endpoint is also currently
unauthenticated; per-user auth + per-UID limiting is the real denial-of-wallet
fix and needs a coordinating client change (Firebase ID token on every call).
"""

import logging
import os
from typing import Optional

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger("jumpr_ai_coach")

# ============================================================
# COST / ABUSE LIMITS
# ============================================================
MAX_OUTPUT_TOKENS = 1024          # hard cap on tokens billed per reply
MAX_MESSAGE_CHARS = 8000          # per-message input cap (~2k tokens)
MAX_HISTORY_MESSAGES = 100        # cap on turns of history replayed
MAX_TOTAL_REQUEST_CHARS = 60000   # cap on total prompt text per request (~15k tokens)
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "60/minute")  # per-client request cap

# Rate limiter keyed on client address. Behind Railway's proxy this may resolve
# to the proxy rather than the true client IP, in which case the limit behaves
# as a global circuit-breaker — still a useful denial-of-wallet backstop, but
# not per-user. Real per-user limiting arrives with authentication.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="JUMPR AI Coach")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mobile clients have no browser Origin, so CORS is not a security control here.
# Credentials disabled: "*" + allow_credentials is an invalid combination and
# is unused by the app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize the Anthropic client
# Set your API key as an environment variable: ANTHROPIC_API_KEY
client = Anthropic()

# ============================================================
# SYSTEM PROMPT - This is what makes the AI a basketball coach
# ============================================================
SYSTEM_PROMPT = """You are the JUMPR AI Basketball Coach — an expert shooting coach built into the JUMPR basketball training app.

YOUR ROLE:
- You help basketball players improve their jump shot form and shooting consistency.
- You explain shooting mechanics in simple, actionable terms.
- You motivate players to stay consistent with their training.
- You reference the player's data when provided (form score, detected issues, drills).

YOUR PERSONALITY:
- Talk like a real basketball coach — confident, encouraging, direct.
- Keep responses concise and practical. Players want quick answers, not essays.
- Use basketball terminology naturally but explain technical concepts simply.
- Be motivating without being corny. Think professional trainer, not hype man.

YOUR KNOWLEDGE AREAS:
- Jump shot biomechanics (base, knee bend, set point, release, follow-through, arc)
- Common shooting errors (elbow flare, low set point, flat arc, thumbing the ball, inconsistent base)
- Drills and exercises to fix specific form issues
- Mental aspects of shooting (confidence, rhythm, consistency)
- Practice structure and training frequency
- Game-situation shooting vs practice shooting

RULES:
- Only discuss basketball shooting and training topics. If asked about unrelated topics, redirect back to basketball training.
- Never make up statistics or cite fake studies.
- If you don't know something, say so and suggest the player consult a trainer.
- Keep responses under 200 words unless the player asks for a detailed breakdown.
- When the player's analysis data is provided, reference their specific issues and scores.

PLAYER CONTEXT (if provided):
When player data is included in the message, use it to personalize your advice. Reference their:
- Form score and what it means
- Specific detected issues
- Assigned drills and progress
- Goals and improvement trends
"""


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str = Field(max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_history: list[ChatMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )
    player_context: Optional[dict] = None  # Optional player data for personalization


class ChatResponse(BaseModel):
    response: str


def _extract_text(response) -> str:
    """Return the first text block, tolerating non-text blocks (e.g. thinking)."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


# ============================================================
# ENDPOINTS
# ============================================================
@app.post("/ai-coach/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(request: Request, body: ChatRequest):
    """
    Main AI Coach chat endpoint.

    Send a message and optional conversation history.
    Returns the AI Coach's response.

    Optional player_context can include:
    {
        "form_score": 72,
        "detected_issues": ["elbow_flare", "flat_arc"],
        "assigned_drills": ["wall_shooting", "one_hand_form"],
        "goal_score": 85,
        "sessions_completed": 5,
        "shooting_hand": "right",
        "position": "guard",
        "experience_level": "intermediate"
    }
    """
    # Bound total prompt size before spending any tokens (denial-of-wallet guard).
    total_chars = len(body.message) + sum(
        len(m.content) for m in body.conversation_history
    )
    if total_chars > MAX_TOTAL_REQUEST_CHARS:
        raise HTTPException(
            status_code=413,
            detail="Conversation is too long. Please start a new chat.",
        )

    # Build the system prompt with player context if provided
    system = SYSTEM_PROMPT
    if body.player_context:
        context_str = "\n\nCURRENT PLAYER DATA:\n"
        ctx = body.player_context

        if "form_score" in ctx:
            context_str += f"- Current form score: {ctx['form_score']}/100\n"
        if "detected_issues" in ctx:
            issues = ", ".join(str(i) for i in ctx["detected_issues"])
            context_str += f"- Detected issues: {issues}\n"
        if "assigned_drills" in ctx:
            drills = ", ".join(str(d) for d in ctx["assigned_drills"])
            context_str += f"- Assigned drills: {drills}\n"
        if "goal_score" in ctx:
            context_str += f"- Goal score: {ctx['goal_score']}/100\n"
        if "sessions_completed" in ctx:
            context_str += f"- Sessions completed: {ctx['sessions_completed']}\n"
        if "shooting_hand" in ctx:
            context_str += f"- Shooting hand: {ctx['shooting_hand']}\n"
        if "position" in ctx:
            context_str += f"- Position: {ctx['position']}\n"
        if "experience_level" in ctx:
            context_str += f"- Experience level: {ctx['experience_level']}\n"

        system += context_str

    # Build messages array with conversation history (drop any bad roles).
    messages = []
    for msg in body.conversation_history:
        if msg.role not in ("user", "assistant"):
            continue
        messages.append({"role": msg.role, "content": msg.content})

    # Add the current user message
    messages.append({"role": "user", "content": body.message})

    # Call Claude API
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=messages,
        )
    except Exception:
        # Log the real error server-side; return a generic message to the client.
        logger.exception("AI Coach upstream call failed")
        raise HTTPException(
            status_code=503,
            detail="The AI Coach is temporarily unavailable. Please try again.",
        )

    assistant_message = _extract_text(response)
    if not assistant_message:
        raise HTTPException(
            status_code=503,
            detail="The AI Coach could not generate a response. Please try again.",
        )

    return ChatResponse(response=assistant_message)


@app.get("/ai-coach/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "JUMPR AI Coach"}


@app.get("/ai-coach/suggestions")
async def get_suggestions():
    """
    Returns suggested conversation starters for the AI Coach UI.
    Gazihan's team can use these to show quick-tap suggestions in the chat.
    """
    return {
        "suggestions": [
            "How do I get more arc on my shot?",
            "Why do I keep missing short?",
            "How can I be more consistent?",
            "What should my pre-shot routine be?",
            "How do I fix my elbow flare?",
            "How often should I practice shooting?",
        ]
    }
