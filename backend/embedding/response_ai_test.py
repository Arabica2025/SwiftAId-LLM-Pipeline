import os
from openai import OpenAI
from google import genai
from pathlib import Path
from dotenv import load_dotenv

import faiss_retrieval as retrieve
import topic_router as router



#load_dotenv(Path(__file__).resolve().parent.parent / ".env")

openAIClient = OpenAI(api_key="OPENAI_API_KEY")

rag_context = retrieve.context

user_task = f"""
{router.user_query}
"""

resp = openAIClient.responses.create(
    model="gpt-5-nano",
    input=[
        {"role": "system", "content": f"You are a medical RAG assistant. Provide a step-by step instrcuctions of procedures refering evidence-grounded statements. summarize major symptoms that the user describes. {rag_context}"},
        {"role": "user", "content": f"USER_TASK:\n{user_task}"},
    ],
)

print(resp.output_text)

genaiclient = genai.Client(api_key="GEMINI_KEY")


prompt = f"""
You are a medical RAG assistant. 
Provide a step-by step instrcuctions of procedures refering evidence-grounded statements. 
summarize major symptoms that the user describes.
{rag_context}

TASK:
{user_task}
"""
genaiResponse = genaiclient.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
)

print(genaiResponse.text)