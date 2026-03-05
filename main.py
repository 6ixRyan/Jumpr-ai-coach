"""
JUMPR AI Coach - FastAPI Backend
Powered by Claude (Anthropic)

This is the AI Coach endpoint for the JUMPR basketball training app.
It receives user messages and returns basketball coaching responses.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic
import os
from typing import Optional

app = FastAPI(title="JUMPR AI Coach")

# Allow requests from your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[ChatMessage] = []
    player_context: Optional[dict] = None  # Optional player data for personalization


class ChatResponse(BaseModel):
    response: str


# ============================================================
# ENDPOINTS
# ============================================================
@app.post("/ai-coach/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
    try:
        # Build the system prompt with player context if provided
        system = SYSTEM_PROMPT
        if request.player_context:
            context_str = "\n\nCURRENT PLAYER DATA:\n"
            ctx = request.player_context

            if "form_score" in ctx:
                context_str += f"- Current form score: {ctx['form_score']}/100\n"
            if "detected_issues" in ctx:
                issues = ", ".join(ctx["detected_issues"])
                context_str += f"- Detected issues: {issues}\n"
            if "assigned_drills" in ctx:
                drills = ", ".join(ctx["assigned_drills"])
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

        # Build messages array with conversation history
        messages = []
        for msg in request.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add the current user message
        messages.append({"role": "user", "content": request.message})

        # Call Claude API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=messages,
        )

        # Extract text response
        assistant_message = response.content[0].text

        return ChatResponse(response=assistant_message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
