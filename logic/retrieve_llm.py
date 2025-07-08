
import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
ASTRA_DB_COLLECTION = "balzi_rossi"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embeddings and vectorstore
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    openai_api_key=OPENAI_API_KEY
)

vectorstore = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=ASTRA_DB_COLLECTION,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

# LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# Custom output parser that returns only the raw text (clean for TTS)
class VoiceOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # We can clean or trim the response here if needed
        return text.strip()

# System prompt template for RAG, incorporating user profile to guide voice style
SYSTEM_PROMPT_TEMPLATE = """
You are an empathetic, multilingual AI assistant for the Balzi Rossi Archaeological Site and Museum.
Your goal is to provide clear, helpful, and engaging answers strictly based on the given context.

User profile:
- Emotion: {emotion}
- Tone: {tone}
- Age group: {age_group}
- Language: {language}

Instructions:

- Always respond in the user’s selected language: '{language}'.
- Adapt style to emotion:
  - 'frustrated' or 'anger': calm, reassuring, concise.
  - 'happy' or 'joy': enthusiastic, warm, positive.
  - 'surprise' or 'curious': detailed, intriguing.
  - others: friendly and balanced.
- Adapt tone:
  - 'demanding' or 'assertive': polite but direct.
  - 'polite' or 'friendly': warm and pleasant.
- Adapt content for age group:
  - 'child': simple, short sentences, analogies.
  - 'teenager': engaging details related to interests.
  - 'adult': clear, informative.
  - 'senior': respectful, clear.
  - 'unknown': treat as adult.

General rules:
- Use only the provided context to answer.
- If not enough info, politely say so.
- Do NOT invent facts.
- If query unrelated to museum, redirect politely.

Context:
{context}

User question:
{question}

Respond concisely and clearly:
"""

def prepare_prompt_input(inputs: dict) -> dict:
    return {
        "question": inputs["question"],
        "context": "\n\n".join(doc.page_content for doc in inputs["context"]),
        "emotion": inputs.get("emotion", "neutral"),
        "tone": inputs.get("tone", "friendly"),
        "age_group": inputs.get("age_group", "adult"),
        "language": inputs.get("language", "en")
    }

def get_rag_chain():
    def retriever_with_lang_filter(input_dict: dict):
        # Filter by user selected language ONLY (no auto detect)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"language": input_dict.get("language", "en")}}
        )
        return {"retriever": retriever}

    rag_chain = (
        RunnablePassthrough.assign(
            retriever_info=RunnableLambda(retriever_with_lang_filter)
        )
        .assign(
            context=lambda x: x["retriever_info"]["retriever"].invoke(x["question"])
        )
        .assign(
            prompt_input=RunnableLambda(prepare_prompt_input)
        )
        .assign(
            answer=ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
                   | llm
                   | VoiceOutputParser()
        )
        .pick("answer")
    )
    return rag_chain

# Example usage
if __name__ == "__main__":
    rag_chain = get_rag_chain()

    test_input = {
        "question": "What can I see in the Balzi Rossi Museum?",
        "emotion": "happy",
        "tone": "friendly",
        "age_group": "adult",
        "language": "en"
    }

    logging.info(f"Generating response for language={test_input['language']}, emotion={test_input['emotion']}, tone={test_input['tone']}, age_group={test_input['age_group']}")
    response = rag_chain.invoke(test_input)
    print("\n--- AI Response for Voice TTS ---")
    print(response)
