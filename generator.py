from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-base"
FALLBACK_ANSWER = "I don't know based on the provided context."


def load_generator_model():
    """
    Load the tokenizer and language model used for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return tokenizer, model


def build_prompt(query: str, retrieved_docs: list[str]) -> str:
    """
    Build the grounded prompt sent to the language model.
    """
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are an Islamic finance assistant.
Answer the question using ONLY the supplied context.

If the answer is not available in the context, reply exactly with:
{FALLBACK_ANSWER}

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt.strip()


def generate_answer(query: str, retrieved_docs: list[str], tokenizer, model) -> str:
    """
    Generate an answer using the retrieved context.
    """
    if not retrieved_docs:
        return FALLBACK_ANSWER

    prompt = build_prompt(query, retrieved_docs)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        min_new_tokens=20,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

    return answer.replace("\n", " ")