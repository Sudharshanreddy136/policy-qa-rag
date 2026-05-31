import os
from groq import Groq

# Get FREE API key from https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your api key ")

client = Groq(api_key=GROQ_API_KEY)


def ask_groq(question: str, context: str, history: str = "", filename: str = "") -> str:
    """
    Send question + FAISS retrieved chunks to Groq Llama3 and return answer.
    """
    system_prompt = f"""You are a helpful assistant that answers questions about company documents and policies.
You have been given relevant sections retrieved from the document: "{filename}"

STRICT RULES:
1. Answer ONLY using the information in the CONTEXT below.
2. If the answer is not in the context, say "I couldn't find that information in the document."
3. Be clear, concise, and well structured.
4. If the context mentions a page number, include it in your answer.
5. Never make up or assume information not present in the context.
6. If asked a follow-up question, use the conversation history to understand context.
"""

    user_message = f"""CONTEXT RETRIEVED FROM DOCUMENT (via FAISS vector search):
{context}

{"PREVIOUS CONVERSATION:" + chr(10) + history if history else ""}

QUESTION: {question}

Answer based only on the context above."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
            max_tokens=512,
            timeout=30,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling Groq API: {str(e)}"
