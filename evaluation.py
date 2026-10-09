from pathlib import Path

from generator import (
    FALLBACK_ANSWER,
    generate_answer,
    load_generator_model,
)
from ingest import initialize_knowledge_base
from retriever import detect_contract, retrieve_context
from test_cases import test_cases


DATA_DIR = Path("data")
TOP_K = 2


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    """
    Return True when every expected keyword appears in the text.
    The comparison is case-insensitive.
    """
    normalized_text = text.lower()

    return all(
        keyword.lower() in normalized_text
        for keyword in keywords
    )


def evaluate_system():
    """
    Run all test cases and evaluate the different RAG stages.
    """
    print("\nInitializing RAG system...")
    collection, embedder = initialize_knowledge_base(DATA_DIR)

    print("\nLoading generator model...")
    tokenizer, model = load_generator_model()

    total_tests = len(test_cases)
    contract_passes = 0
    retrieval_passes = 0
    retrieval_tests = 0
    generation_passes = 0
    overall_passes = 0

    print("\n" + "=" * 60)
    print("RAG EVALUATION")
    print("=" * 60)

    for test in test_cases:
        test_id = test["id"]
        question = test["question"]
        expected_contract = test["expected_contract"]
        answerable = test["answerable"]

        print(f"\nTest: {test_id}")
        print(f"Question: {question}")

        # -----------------------------------------
        # 1. Evaluate contract detection
        # -----------------------------------------
        detected_contract = detect_contract(question)

        contract_pass = detected_contract == expected_contract

        if contract_pass:
            contract_passes += 1

        print(f"Expected contract: {expected_contract}")
        print(f"Detected contract: {detected_contract}")
        print(
            "Contract detection:",
            "PASS" if contract_pass else "FAIL",
        )

        # Default values for questions with no supported contract
        retrieved_docs = []
        retrieved_metadatas = []
        retrieved_distances = []

        # -----------------------------------------
        # 2. Run retrieval when a contract is found
        # -----------------------------------------
        if detected_contract is not None:
            retrieval_result = retrieve_context(
                query=question,
                collection=collection,
                embedder=embedder,
                top_k=TOP_K,
                contract=detected_contract,
            )

            retrieved_docs = retrieval_result["documents"]
            retrieved_metadatas = retrieval_result["metadatas"]
            retrieved_distances = retrieval_result["distances"]

            print("\nRetrieved chunks:")

            for rank, document in enumerate(retrieved_docs):
                metadata = retrieved_metadatas[rank]
                distance = retrieved_distances[rank]

                print(
                    f"  Rank {rank + 1}: "
                    f"chunk {metadata['chunk_id']}, "
                    f"distance {distance:.4f}"
                )
                print(f"  {document[:180]}...")

        # -----------------------------------------
        # 3. Evaluate retrieval for answerable tests
        # -----------------------------------------
        if answerable:
            retrieval_tests += 1
            combined_context = "\n".join(retrieved_docs)

            retrieval_pass = contains_all_keywords(
                combined_context,
                test["evidence_keywords"],
            )

            if retrieval_pass:
                retrieval_passes += 1

            print(
                "\nRetrieval evidence:",
                "PASS" if retrieval_pass else "FAIL",
            )
        else:
            retrieval_pass = None
            print("\nRetrieval evidence: NOT APPLICABLE")

        # -----------------------------------------
        # 4. Generate an answer
        # -----------------------------------------
        if detected_contract is None:
            answer = FALLBACK_ANSWER
        else:
            answer = generate_answer(
                query=question,
                retrieved_docs=retrieved_docs,
                tokenizer=tokenizer,
                model=model,
            )

        print(f"Generated answer: {answer}")

        # -----------------------------------------
        # 5. Evaluate generation
        # -----------------------------------------
        if answerable:
            generation_pass = contains_all_keywords(
                answer,
                test["answer_keywords"],
            )
        else:
            generation_pass = (
                answer.strip().lower()
                == FALLBACK_ANSWER.strip().lower()
            )

        if generation_pass:
            generation_passes += 1

        print(
            "Generation:",
            "PASS" if generation_pass else "FAIL",
        )

        # -----------------------------------------
        # 6. Calculate overall result
        # -----------------------------------------
        if answerable:
            overall_pass = (
                contract_pass
                and retrieval_pass
                and generation_pass
            )
        else:
            overall_pass = contract_pass and generation_pass

        if overall_pass:
            overall_passes += 1

        print(
            "Overall result:",
            "PASS" if overall_pass else "FAIL",
        )
        print("-" * 60)

    # -----------------------------------------
    # 7. Print evaluation summary
    # -----------------------------------------
    print("\nEVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Contract detection: "
        f"{contract_passes}/{total_tests}"
    )

    print(
        f"Retrieval evidence: "
        f"{retrieval_passes}/{retrieval_tests}"
    )

    print(
        f"Generation: "
        f"{generation_passes}/{total_tests}"
    )

    print(
        f"Overall: "
        f"{overall_passes}/{total_tests}"
    )


if __name__ == "__main__":
    """
    Load the knowledge base → load the model → loop through test cases
     → detect the contract → retrieve chunks → check evidence 
     → generate an answer → evaluate the answer → calculate totals.
    """
    evaluate_system()