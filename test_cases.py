test_cases = [
    {
        "id": "murabaha_ownership",
        "question": (
            "In Murabaha, must the seller own the asset "
            "before selling it?"
        ),
        "expected_contract": "murabaha",
        "answerable": True,
        "evidence_keywords": [
            "own",
            "possess",
            "asset",
        ],
        "answer_keywords": [
            "own",
            "asset",
        ],
    },
    {
        "id": "ijara_definition",
        "question": "What is Ijara?",
        "expected_contract": "ijara",
        "answerable": True,
        "evidence_keywords": [
            "lease",
            "asset",
            "owner",
        ],
        "answer_keywords": [
            "lease",
            "asset",
        ],
    },
    {
        "id": "sukuk_definition",
        "question": "What is Sukuk?",
        "expected_contract": "sukuk",
        "answerable": True,
        "evidence_keywords": [
            "ownership",
            "asset",
        ],
        "answer_keywords": [
            "ownership",
            "asset",
        ],
    },
    {
        "id": "sukuk_unsupported_statistic",
        "question": (
            "What was the total global Sukuk issuance "
            "volume in 2025?"
        ),
        "expected_contract": "sukuk",
        "answerable": False,
        "evidence_keywords": [],
        "answer_keywords": [],
    },
    {
        "id": "unsupported_contract",
        "question": "What is Musharakah?",
        "expected_contract": None,
        "answerable": False,
        "evidence_keywords": [],
        "answer_keywords": [],
    },
]