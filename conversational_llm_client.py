"""
Implement a client for interacting with OpenAI's API, specifically for generating responses in a conversational context.
This client maintains conversation history and allows for context to be provided to the model.
The client uses the OpenAI Python SDK and requires an API key to be set in the environment variables.
"""

from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# FIX:
# Create client once instead of recreating it for every request.
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OpenAI API key not found. Please set it in the .env file."
    )

client = OpenAI(api_key=api_key)


def generate_response(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    context: str,
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate a response using OpenAI and maintain conversation history.
    """

    # System Prompt
    system_prompt = (
        "You are an assistant that provides information "
        "about NASA's missions, research, and space exploration."
    )

    # ============================================================
    # FIX 1:
    # Do NOT overwrite conversation_history.
    #
    # Your original code discarded all previous messages.
    #
    # Instead create a separate 'messages' list that will be sent
    # to the model.
    # ============================================================

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },

        # FIX 2:
        # "context" is NOT a valid message field.
        #
        # Original:
        # {
        #     "role": "system",
        #     "content": system_prompt,
        #     "context": context
        # }
        #
        # Context should be sent as content.
        {
            "role": "system",
            "content": context
        }
    ]

    # ============================================================
    # FIX 3:
    # Add previous conversation history AFTER system messages.
    #
    # Message order matters.
    # ============================================================

    messages.extend(conversation_history)

    # ============================================================
    # FIX 4:
    # Add current user message.
    #
    # Original code tried to include an assistant reply before
    # it existed.
    # ============================================================

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    # ============================================================
    # FIX 5:
    # Use the model parameter instead of hardcoding.
    # ============================================================

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=50
    )

    # Extract reply
    assistant_reply = response.choices[0].message.content

    # ============================================================
    # FIX 6:
    # Save BOTH user and assistant messages to history.
    #
    # Otherwise future requests lose context.
    # ============================================================

    conversation_history.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    conversation_history.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )

    return assistant_reply