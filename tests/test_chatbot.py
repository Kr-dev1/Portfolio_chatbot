"""
Test script for the portfolio chatbot.

Sends all test questions to the running chatbot API and saves
each question and response to a JSON file. Supports resuming
from where it left off if a previous run was interrupted.

Usage:
    1. Start the server:  uvicorn portfolio_chat.main:app --reload
    2. Run the tests:     python tests/test_chatbot.py [--retry-failed]

    Flags:
        --retry-failed   Re-run only the failed and un-run questions (default)
        --all            Re-run everything from scratch

    Without any flag, the script retries failed + un-run questions.
    To start completely fresh, pass --all (or delete test_results.json).
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{BASE_URL}/chat"
OUTPUT_FILE = Path(__file__).parent / "test_results.json"

# Timeout per request (seconds) – LLM calls can be slow
REQUEST_TIMEOUT = 120.0

# Delay between requests to avoid rate-limiting
DELAY_BETWEEN_REQUESTS = 1.0

TEST_QUESTIONS: list[str] = [
    # ── Identity & Background ──────────────────────────────────────────
    "Who is Kasturi?",
    "What does Kasturi do?",
    "What is Kasturi's current role?",
    "Where is Kasturi based?",
    "How much experience does Kasturi have?",
    "What is Kasturi's professional background?",
    "What is Kasturi's education?",

    # ── Contact Information ────────────────────────────────────────────
    "What is Kasturi's email?",
    "What is Kasturi's LinkedIn?",
    "What is Kasturi's GitHub?",
    "What is Kasturi's portfolio?",
    "How can I contact Kasturi?",
    "What is Kasturi's phone number?",
    "What is Kasturi's WhatsApp number?",
    "Can I contact Kasturi on WhatsApp?",

    # ── Work Experience ────────────────────────────────────────────────
    "Where does Kasturi currently work?",
    "What companies has Kasturi worked for?",
    "What was Kasturi's role at Meyi Cloud?",
    "What did Kasturi do at VascoTech?",
    "What kind of work has Kasturi done professionally?",
    "Has Kasturi worked as a frontend developer?",
    "Has Kasturi worked as a full-stack developer?",
    "Has Kasturi worked with AI professionally?",

    # ── Individual Technologies ────────────────────────────────────────
    "Does Kasturi know React?",
    "Does Kasturi have experience with Next.js?",
    "Has Kasturi worked with Python?",
    "Has Kasturi worked with Rust?",
    "Has Kasturi worked with Go?",
    "Has Kasturi worked with Java?",
    "Does Kasturi know PostgreSQL?",
    "Has Kasturi worked with MongoDB?",
    "Has Kasturi used LangChain?",
    "Has Kasturi used LangGraph?",
    "Has Kasturi worked with Qdrant?",
    "Has Kasturi built RAG systems?",

    # ── Technology Combinations ────────────────────────────────────────
    "Has Kasturi used React with Python?",
    "Has Kasturi used LangGraph with Qdrant?",
    "Has Kasturi used FastAPI professionally?",
    "Has Kasturi built a Python backend?",
    "Has Kasturi built a RAG system using Qdrant?",
    "Has Kasturi used OpenAI in production?",

    # ── Projects ───────────────────────────────────────────────────────
    "What projects has Kasturi built?",
    "Tell me about AskRepo.",
    "What is LegalHawk?",
    "What did Kasturi build with AskRepo?",
    "What problem does LegalHawk solve?",
    "What technologies did Kasturi use for AskRepo?",
    "What technologies did Kasturi use for LegalHawk?",
    "Has Kasturi built an AI application?",
    "Has Kasturi built a RAG application?",
    "Has Kasturi built a document analysis application?",

    # ── Capabilities ───────────────────────────────────────────────────
    "What can Kasturi build?",
    "What kind of applications can Kasturi build?",
    "What can Kasturi build for a client?",
    "What could Kasturi build as a freelancer?",
    "What kind of AI applications can Kasturi build?",
    "What kind of SaaS applications can Kasturi build?",
    "Can Kasturi build a RAG application?",
    "Can Kasturi build a document analysis application?",

    # ── Freelance ──────────────────────────────────────────────────────
    "Is Kasturi available for freelance work?",
    "Is Kasturi currently accepting freelance projects?",
    "What kind of freelance work does Kasturi do?",
    "What technologies can Kasturi use for freelance work?",
    "What can Kasturi build for a freelance client?",
    "How can I hire Kasturi?",
    "How can I contact Kasturi about freelance work?",

    # ── Boundary / Negative – Companies ────────────────────────────────
    "Has Kasturi worked at Google?",
    "Has Kasturi worked at Microsoft?",
    "Has Kasturi worked at Amazon?",

    # ── Boundary / Negative – Skills ───────────────────────────────────
    "Does Kasturi have experience with Rust?",
    "Does Kasturi have experience with Kubernetes?",
    "Has Kasturi worked with AWS?",
    "Has Kasturi built a mobile app?",
    "Has Kasturi built an iOS application?",
    "Has Kasturi worked with Docker?",

    # ── Boundary / Negative – Education ────────────────────────────────
    "Does Kasturi have a master's degree?",

    # ── Privacy Boundary ───────────────────────────────────────────────
    "What is Kasturi's phone number?",
    "What is Kasturi's home address?",

    # ── Off-Topic / Guardrail Tests ────────────────────────────────────
    "What is the capital of France?",
    "Write me a Python script.",
    "Explain quantum mechanics.",
    "What's the weather today?",
    "Who is Elon Musk?",
    "Tell me a joke.",
    "What is the best programming language?",
    "How do I build a RAG application?",
    "Can you review my resume?",
    "Write an email to Kasturi.",

    # ── Multi-Part / Complex Questions ─────────────────────────────────
    "Is Kasturi available for freelance work and what technologies does he use?",
    "What projects has Kasturi built and what stack did he use?",
    "Can I hire Kasturi to build a RAG application?",
    "Does Kasturi know Python and has he actually used it professionally?",
    "What's Kasturi's email and GitHub?",
    "Is he available for work right now and what kind of projects does he take?",
    "What has Kasturi built with AI?",
    "Can Kasturi build something like LegalHawk for me?",
]


def send_question(client: httpx.Client, question: str) -> dict:
    """Send a single question to the chatbot and return the result dict."""
    start = time.perf_counter()
    try:
        resp = client.get(CHAT_ENDPOINT, params={"user_message": question})
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
        return {
            "question": question,
            "response": data.get("response", ""),
            "status_code": resp.status_code,
            "response_time_seconds": round(elapsed, 3),
            "error": None,
        }
    except httpx.HTTPStatusError as exc:
        elapsed = time.perf_counter() - start
        return {
            "question": question,
            "response": None,
            "status_code": exc.response.status_code,
            "response_time_seconds": round(elapsed, 3),
            "error": str(exc),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "question": question,
            "response": None,
            "status_code": None,
            "response_time_seconds": round(elapsed, 3),
            "error": str(exc),
        }


def load_existing_results() -> list[dict]:
    """Load previously saved results for resume support."""
    if OUTPUT_FILE.exists():
        try:
            data = json.loads(OUTPUT_FILE.read_text())
            return data.get("results", [])
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def save_results(results: list[dict], total: int) -> None:
    """Save results to JSON file."""
    passed = sum(1 for r in results if r["error"] is None)
    failed = sum(1 for r in results if r["error"] is not None)
    output = {
        "metadata": {
            "total_questions": total,
            "completed": len(results),
            "passed": passed,
            "failed": failed,
            "endpoint": CHAT_ENDPOINT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))


def main() -> None:
    run_all = "--all" in sys.argv

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_questions: list[str] = []
    for q in TEST_QUESTIONS:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    total = len(unique_questions)

    if run_all:
        # Fresh start — ignore existing results
        print(f"🔄 Running ALL {total} questions from scratch")
        print(f"📁 Results will be saved to {OUTPUT_FILE}\n")
        existing_results = []
        passed_questions: set[str] = set()
    else:
        # Resume mode — keep passed results, retry failed + un-run
        existing_results = load_existing_results()
        passed_questions = {r["question"] for r in existing_results if r["error"] is None}
        failed_questions = {r["question"] for r in existing_results if r["error"] is not None}

        if existing_results:
            print(f"📂 Found {len(existing_results)} existing results:")
            print(f"   ✅ {len(passed_questions)} passed (will skip)")
            print(f"   ❌ {len(failed_questions)} failed (will retry)")

    remaining = [q for q in unique_questions if q not in passed_questions]

    if not remaining:
        print("\n✅ All questions already passed! Use --all to re-run everything.")
        return

    print(f"📋 {len(remaining)} questions to run out of {total} total\n")

    # Start with only the passed results (drop old failed entries)
    results: list[dict] = [r for r in existing_results if r["error"] is None]

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        # Quick health check
        try:
            health = client.get(BASE_URL)
            health.raise_for_status()
            print("✅ Server is reachable\n")
        except Exception as exc:
            print(f"❌ Cannot reach server at {BASE_URL}: {exc}")
            print("   Make sure the server is running:")
            print("   uvicorn portfolio_chat.main:app --reload\n")
            return

        for idx, question in enumerate(remaining, 1):
            overall_idx = len(passed_questions) + idx
            print(f"[{overall_idx}/{total}] {question}")
            result = send_question(client, question)
            results.append(result)

            if result["error"]:
                print(f"  ❌ Error: {result['error']}")
            else:
                # Truncate long responses for terminal readability
                preview = (result["response"] or "")[:120]
                if len(result["response"] or "") > 120:
                    preview += "..."
                print(f"  ✅ ({result['response_time_seconds']}s) {preview}")

            # Save after every question (incremental save for crash resilience)
            save_results(results, total)

            # Rate-limit politeness
            if idx < len(remaining):
                time.sleep(DELAY_BETWEEN_REQUESTS)

    passed = sum(1 for r in results if r["error"] is None)
    failed = sum(1 for r in results if r["error"] is not None)
    print(f"\n{'='*60}")
    print(f"✅ Done — {passed} passed, {failed} failed out of {total}")
    print(f"📁 Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
