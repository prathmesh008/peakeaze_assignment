from __future__ import annotations

import contextvars
import json
import logging
import os
from enum import Enum
from functools import wraps
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


_client: genai.Client | None = None
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Obtain one at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


class IntentCategory(str, Enum):
    TECHNICAL_SUPPORT = "Technical Support"
    BILLING_ISSUE = "Billing Issue"
    GENERAL_FEEDBACK = "General Feedback"


class IntentRouting(BaseModel):
    intent: IntentCategory = Field(
        description="The classified intent category of the customer message."
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen."
    )


class TroubleshootingStep(BaseModel):
    step_number: int = Field(description="Ordinal position of this step.")
    instruction: str = Field(description="Clear, actionable instruction.")


class TechnicalSupportResult(BaseModel):
    entities: list[str] = Field(
        description="Software or hardware product names mentioned."
    )
    problem_summary: str = Field(
        description="One-sentence summary of the reported issue."
    )
    troubleshooting_steps: list[TroubleshootingStep] = Field(
        description="Ordered list of troubleshooting steps."
    )


class BillingIssueResult(BaseModel):
    monetary_values: list[str] = Field(
        description="Dollar amounts or monetary figures mentioned."
    )
    dates_mentioned: list[str] = Field(
        description="Relevant dates or time-frames referenced."
    )
    urgency: str = Field(
        description="Urgency level: 'high', 'medium', or 'low'."
    )
    urgency_reasoning: str = Field(
        description="Explanation of the urgency classification."
    )


class GeneralFeedbackResult(BaseModel):
    sentiment: str = Field(
        description="Overall sentiment: 'positive', 'neutral', or 'negative'."
    )
    key_topics: list[str] = Field(
        description="Main topics or themes in the feedback."
    )
    automated_response_draft: str = Field(
        description="A polite, professional automated reply to the customer."
    )


class WorkflowResult(BaseModel):
    intent: IntentCategory
    intent_reasoning: str
    details: dict[str, Any]
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the pipeline output (0–1).",
    )


class ConfidenceEvaluation(BaseModel):
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    confidence_reasoning: str = Field(
        description="Brief rationale for the assigned confidence score."
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_gemini(
    prompt: str,
    response_schema: type[BaseModel],
) -> dict[str, Any]:
    client = _get_client()

    logger.info("Calling Gemini model=%s schema=%s", MODEL_NAME, response_schema.__name__)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.2,
        ),
    )

    parsed: dict[str, Any] = json.loads(response.text)
    logger.debug("Raw response: %s", parsed)
    return parsed


_current_steps = contextvars.ContextVar("current_steps")


def track_step(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("Executing step: %s", name)
            result = func(*args, **kwargs)
            try:
                _current_steps.get().append({"step": name, "output": result})
            except LookupError:
                pass
            return result
        return wrapper
    return decorator


@track_step("routing")
def _step1_route_intent(text: str) -> dict[str, Any]:
    prompt = (
        "You are a customer-support triage agent. "
        "Analyze the following customer message and classify it into exactly one "
        "of these categories: 'Technical Support', 'Billing Issue', or 'General Feedback'.\n\n"
        f"Customer message:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Return the classification along with a brief reasoning."
    )
    return _call_gemini(prompt, IntentRouting)


@track_step("extraction")
def _step2_technical_support(text: str) -> dict[str, Any]:
    prompt = (
        "You are a senior technical support engineer. "
        "Analyze the following customer message and:\n"
        "1. Extract every software or hardware product name mentioned.\n"
        "2. Summarize the reported problem in one sentence.\n"
        "3. Provide an ordered list of practical troubleshooting steps.\n\n"
        f"Customer message:\n\"\"\"\n{text}\n\"\"\"\n"
    )
    return _call_gemini(prompt, TechnicalSupportResult)


@track_step("extraction")
def _step2_billing_issue(text: str) -> dict[str, Any]:
    prompt = (
        "You are a billing support specialist. "
        "Analyze the following customer message and:\n"
        "1. Extract all monetary values or dollar amounts mentioned.\n"
        "2. Extract all dates or time-frames referenced.\n"
        "3. Classify the urgency as 'high', 'medium', or 'low'.\n"
        "4. Explain why you chose that urgency level.\n\n"
        f"Customer message:\n\"\"\"\n{text}\n\"\"\"\n"
    )
    return _call_gemini(prompt, BillingIssueResult)


@track_step("extraction")
def _step2_general_feedback(text: str) -> dict[str, Any]:
    prompt = (
        "You are a customer experience analyst. "
        "Analyze the following customer feedback and:\n"
        "1. Determine the overall sentiment ('positive', 'neutral', or 'negative').\n"
        "2. List the main topics or themes in the feedback.\n"
        "3. Draft a polite, professional automated response to the customer.\n\n"
        f"Customer message:\n\"\"\"\n{text}\n\"\"\"\n"
    )
    return _call_gemini(prompt, GeneralFeedbackResult)


@track_step("confidence")
def _step3_confidence(text: str, intent: str, details: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "You are a quality-assurance reviewer for a customer-support AI system. "
        "Given the original customer message, the classified intent, and the extracted details, "
        "assign a confidence score between 0.0 and 1.0 that reflects how accurate and complete "
        "the pipeline output is. Consider whether the intent seems correct and whether the "
        "extracted details are relevant and non-hallucinated.\n\n"
        f"Original message:\n\"\"\"\n{text}\n\"\"\"\n\n"
        f"Classified intent: {intent}\n\n"
        f"Extracted details:\n{details}\n\n"
        "Return the confidence score and a brief reasoning."
    )
    return _call_gemini(prompt, ConfidenceEvaluation)


_INTENT_HANDLERS: dict[str, Any] = {
    IntentCategory.TECHNICAL_SUPPORT.value: _step2_technical_support,
    IntentCategory.BILLING_ISSUE.value: _step2_billing_issue,
    IntentCategory.GENERAL_FEEDBACK.value: _step2_general_feedback,
}


def run_workflow(text: str) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    token = _current_steps.set(steps)

    try:
        routing = _step1_route_intent(text)
        intent = routing["intent"]

        handler = _INTENT_HANDLERS.get(intent)
        if handler is None:
            raise ValueError(f"Unknown intent returned by the model: {intent}")

        details = handler(text)
        confidence = _step3_confidence(text, intent, details)

        result = WorkflowResult(
            intent=intent,
            intent_reasoning=routing["reasoning"],
            details=details,
            confidence_score=confidence["confidence_score"],
        )
        return {"result": result.model_dump(), "steps": steps}
    finally:
        _current_steps.reset(token)
