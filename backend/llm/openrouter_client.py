import os
import sys
import csv
import time
import random
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()

MODEL = "qwen/qwen3.6-27b"

MODEL_REASONING = {
    "qwen/qwen3.6-27b": {"enabled": False},
    "openai/gpt-5.4-mini": None, # non-reasoning is default
    "mistralai/mistral-medium-3-5": None,
}

BASE_URL = "https://openrouter.ai/api/v1"
NUM_QUESTIONS = 5

TEST_DATASET_PATH = "../../archive/quiz_questions.csv"
TEST_RESULTS_PATH = "../../results"

# Client initialization
client = None

def get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in your .env file.")
        client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return client

# Read the reasoning text out of a response message
def extract_reasoning(msg) -> str:
    direct = getattr(msg, "reasoning", None)
    if direct:
        return direct if isinstance(direct, str) else str(direct)

    details = getattr(msg, "reasoning_details", None)
    if details:
        texts = [d.get("text", "") for d in details if isinstance(d, dict)]
        return "\n".join(t for t in texts if t)

    return ""

# def reasoning_config(reasoning_effort: str | None) -> dict:
#     return {"enabled": False} if reasoning_effort is None else {"effort": reasoning_effort}

def reasoning_config(reasoning_effort: str | None, model: str) -> dict | None:
    if reasoning_effort is not None:
        return {"effort": reasoning_effort}
    return MODEL_REASONING.get(model, {"enabled": False})

# Reasoning disabled by default
# genration doesn't benefit from deliberation and needs to be fast
def generate_question(topic: str, question_number: int, previous_questions: list[str], reasoning_effort: str | None = None, model: str = MODEL) -> tuple[str, str, float]:

    if previous_questions:
        avoid = "\n".join(f"- {q}" for q in previous_questions)
        avoid_text = f"\n\nAlready asked (do not repeat):\n{avoid}"
    else:
        avoid_text = ""

    t_start = time.time()

    kwargs = {}
    cfg = reasoning_config(reasoning_effort, model)
    if cfg is not None:
        kwargs["extra_body"] = {"reasoning": cfg}

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Role: You are a quiz master creating questions for a spoken quiz game.\n\n"
                    "Instructions: Generate exactly one quiz question with a single, clear, verifiable correct answer.\n\n"
                    "Steps:\n"
                    "1. Choose a knowledge domain (history, geography, science, culture, sport, technology, nature, ...).\n"
                    "2. Select one specific, well-established fact within that domain.\n"
                    "3. Phrase it as one natural spoken question.\n\n"
                    "End goal: The player hears the question read aloud and answers by speaking, so it must be immediately understandable without any visual aid.\n\n"
                    "Narrowing: One sentence. No preamble, numbering, or explanation, output the question text only."
                )
            },
            {
                "role": "user",
                "content": f"Create quiz question number {question_number} on the topic: {topic}.{avoid_text}"
            }
        ],
        temperature=0.7,
        max_tokens=1000,
        **kwargs,
        # extra_body={"reasoning": reasoning_config(reasoning_effort)},
    )

    msg = response.choices[0].message
    question = (msg.content or "").strip()
    reasoning = extract_reasoning(msg)
    latency_ms = (time.time() - t_start) * 1000

    if not question:
        raise RuntimeError(
            f"Empty question from {model} "
            f"(finish_reason={response.choices[0].finish_reason})"
        )

    return question, reasoning, latency_ms


# Reasoning disabled by default ("low", "medium" showd now accuracy improvement over "none")
def evaluate_answer(question: str, user_answer: str, reasoning_effort: str | None = None, model: str = MODEL) -> tuple[bool, str, str, float]:

    t_start = time.time()

    kwargs = {}
    cfg = reasoning_config(reasoning_effort, model)
    if cfg is not None:
        kwargs["extra_body"] = {"reasoning": cfg}

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quiz evaluator. Your job is to judge whether a user's answer "
                    "to a quiz question is correct.\n\n"
                    "Rules:\n"
                    "- Be generous: accept answers that are essentially correct even if "
                    "  phrased differently, abbreviated, or slightly misspelled\n"
                    "- The FIRST word of your response must be either CORRECT or WRONG — "
                    "  this is critical, do not start with anything else\n"
                    "- After that first word, add one short sentence of explanation\n"
                    "- If the user did not provide an answer, says they don't know, or "
                    "  gives an unrelated/nonsense response, mark it WRONG and "
                    "  ALWAYS include the correct answer in your explanation\n"
                    "- Keep your total response under 30 words"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"User's answer: {user_answer}\n\n"
                    f"Is this answer correct?"
                )
            }
        ],
        temperature=0.0,
        max_tokens=2000,
        **kwargs,
        # extra_body={"reasoning": reasoning_config(reasoning_effort)},
    )

    msg = response.choices[0].message
    explanation = (msg.content or "").strip()
    reasoning = extract_reasoning(msg)
    latency_ms = (time.time() - t_start) * 1000

    if not explanation:
        return False, "I could not evaluate that answer. Let's continue.", reasoning, latency_ms

    is_correct = explanation.upper().startswith("CORRECT")

    return is_correct, explanation, reasoning, latency_ms

# Terminal quiz mode: py openrouter_client.py
def run_quiz(model: str = MODEL):
    print("\n" + "-" * 50)
    print("LLM: OpenRouter /", model)
    print("-" * 50)

    topic = input("\nEnter a topic (or press Enter for 'general knowledge'): ").strip() \
            or "general knowledge"

    num_input = input(f"How many questions? (press Enter for {NUM_QUESTIONS}): ").strip()
    try:
        num_questions = int(num_input) if num_input else NUM_QUESTIONS
    except ValueError:
        num_questions = NUM_QUESTIONS

    print(f"\nStarting quiz: {num_questions} questions about '{topic}'")
    print("-" * 50)

    correct_count = 0
    previous_questions = []

    for i in range(1, num_questions + 1):

        print(f"\nGenerating question {i} of {num_questions}...")
        question, _, gen_ms = generate_question(topic, i, previous_questions, model=model)
        previous_questions.append(question)
        print(f"\nQuestion {i} ({gen_ms:.0f} ms): {question}")

        user_answer = input("Your answer: ").strip()

        if not user_answer:
            print("(no answer given — marked as wrong)")
            continue

        print("Evaluating...")
        is_correct, explanation, _, eval_ms = evaluate_answer(question, user_answer, model=model)

        if is_correct:
            correct_count += 1
        print(f"{explanation}  ({eval_ms:.0f} ms)")

        print("-" * 50)

    print(f"\n{'-' * 50}")
    print(f"Quiz complete!")
    print(f"Your score: {correct_count} / {num_questions}")
    print(f"{'-' * 50}\n")

# Dataset benchmark 
def answer_question(question: str, reasoning_effort: str | None = None, model: str = MODEL) -> tuple[str, str, float, str]:

    t_start = time.time()

    kwargs = {}
    cfg = reasoning_config(reasoning_effort, model)
    if cfg is not None:
        kwargs["extra_body"] = {"reasoning": cfg}
 
    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a trivia expert answering True/False questions. "
                    "Rules:\n"
                    "- Reply with exactly one word: either True or False\n"
                    "- No explanation, no punctuation, nothing else\n"
                    "- Examples: 'True' or 'False'"
                    # "Role: You are a quiz evaluator judging spoken answers in a voice quiz game.\n\n"
                    # "Instructions: Decide whether the user's answer to the quiz question is correct, and give brief spoken feedback.\n\n"
                    # "Steps:\n"
                    # "1. Identify the correct answer to the question.\n"
                    # "2. Compare the user's answer to it by meaning, not by exact wording.\n"
                    # "3. Decide CORRECT or WRONG.\n"
                    # "4. Write one short sentence of feedback, including the correct answer whenever the user is wrong.\n\n"
                    # "End goal: The user hears your response read aloud, so it must be short, clear, and understandable without any visual aid.\n\n"
                    # "Narrowing:\n"
                    # "- The FIRST word of your response must be either CORRECT or WRONG\n"
                    # "- Be generous: accept answers that are essentially correct even if phrased differently, abbreviated, or slightly misspelled.\n"
                    # "- The answer arrives from speech recognition, so accept phonetically close or lightly garbled words when the intent is clear.\n"
                    # "- If the user gives no answer, says they don't know, or responds with something unrelated, mark it WRONG and state the correct answer.\n"
                    # "- Keep your total response under 30 words."
                )
            },
            {
                "role": "user", 
                "content": question
            }
        ],
        temperature=0.0,
        max_tokens=2000,
        **kwargs,
        # extra_body={"reasoning": reasoning_config(reasoning_effort)},
    )
 
    msg = response.choices[0].message
    answer = (msg.content or "").strip()
    latency_ms = (time.time() - t_start) * 1000
 
    return answer, latency_ms
 
def load_sample(csv_path = TEST_DATASET_PATH, num_questions: int = 50, difficulty = None, category = None) -> list[dict]:

    print(f"Loading questions from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: File {csv_path} not found")
        return
    
    rows = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Total questions in dataset: {len(rows)}")
 
    if difficulty:
        rows = [r for r in rows if r.get("difficulty", "").lower() == difficulty.lower()]
        print(f"After difficulty filter ({difficulty}): {len(rows)} questions")
 
    if category:
        rows = [r for r in rows if r.get("category", "").lower() == category.lower()]
        print(f"After category filter ({category}): {len(rows)} questions")
 
    random.seed(42)
    sampled = random.sample(rows, min(num_questions, len(rows)))
    print(f"Testing on {len(sample)} questions with model: {MODEL}\n")
 
    sample = [{
        "question":       r.get("question", "").strip(),
        "correct_answer": r.get("correct_answer", "").strip(),
        "difficulty":     r.get("difficulty", "").strip(),
        "category":       r.get("category", "").strip(),
    } for r in sampled]
 
    return sample

def run_dataset_test(csv_path = TEST_DATASET_PATH, num_questions = 50, difficulty = None, category = None, model = MODEL):

    safe_model = model.replace("/", "_").replace(":", "_")
    output_path = f"{TEST_RESULTS_PATH}/test_results_{safe_model}.csv"

    # Load csv
    print(f"Loading questions from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: File {csv_path} not found")
        return
    
    rows = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Total questions in dataset: {len(rows)}")

    # Filter difficulty and category
    if difficulty:
        rows = [r for r in rows if r.get("difficulty", "").lower() == difficulty.lower()]
        print(f"After difficulty filter ({difficulty}): {len(rows)} questions")

    if category:
        rows = [r for r in rows if r.get("category", "").lower() == category.lower()]
        print(f"After category filter ({category}): {len(rows)} questions")

    # Sample
    random.seed(42)
    sample = random.sample(rows, min(num_questions, len(rows)))
    print(f"Testing on {len(sample)} questions with model: {MODEL}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run test
    results = []
    correct_count = 0

    for i, row in enumerate(sample, 1):
        question = row.get("question", "").strip()
        correct_answer = row.get("correct_answer", "").strip()
        difficulty_val = row.get("difficulty", "").strip()
        category_val = row.get("category", "").strip()

        print(f"[{i}/{len(sample)}] {question}")
        print(f" Correct answer: {correct_answer}")

        llm_answer, latency_ms = answer_question(question, model=model)
        print(f"LLM answer: {llm_answer} ({latency_ms:.0f}ms)")

        is_correct = llm_answer.strip().lower() == correct_answer.strip().lower()
        
        if is_correct:
            correct_count += 1
            print("Result: CORRECT")
        else:
            print(f"Result: WRONG (expected: {correct_answer})")

        print()

        results.append({
            "question":       question,
            "correct_answer": correct_answer,
            "llm_answer":     llm_answer,
            "is_correct":     is_correct,
            "latency_ms":     round(latency_ms),
            "difficulty":     difficulty_val,
            "category":       category_val,
            "model":          model,
        })

    # Summary
    accuracy = correct_count / len(sample) * 100
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print("-" * 50)
    print(f"Model:     {model}")
    print(f"Questions: {len(sample)}")
    print(f"Correct:   {correct_count} / {len(sample)}")
    print(f"Accuracy:  {accuracy:.1f}%")
    print(f"Avg. latency: {avg_latency:.0f}ms")
    print("-" * 50)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")

# def run_dataset_test_single(sample: list[dict], reasoning_effort: str | None, output_path: str, pause_s: float = 1.0, model: str = MODEL) -> list[dict]:

#     label = "none" if reasoning_effort is None else reasoning_effort
#     print(f"\n{'-' * 50}")
#     print(f"  reasoning_effort = {label}   ({len(sample)} questions)")
#     print(f"{'-' * 50}")
 
#     results = []
#     correct_count = 0
#     empty_count = 0
 
#     for i, row in enumerate(sample, 1):
#         question = row["question"]
#         correct_answer = row["correct_answer"]
 
#         answer, reasoning, latency_ms, finish = answer_question(question, reasoning_effort, model=model)
 
#         is_empty = (answer == "")
#         is_correct = (not is_empty) and answer.strip().lower() == correct_answer.strip().lower()
 
#         if is_empty:
#             empty_count += 1
#         if is_correct:
#             correct_count += 1
 
#         tag = "EMPTY " if is_empty else ("CORRECT" if is_correct else "WRONG  ")
#         print(f"[{i:2d}/{len(sample)}] {tag} | {latency_ms:6.0f} ms | think {len(reasoning):5d} ch | {question[:55]}")
 
#         results.append({
#             "question":         question,
#             "correct_answer":   correct_answer,
#             "llm_answer":       answer,
#             "is_correct":       is_correct,
#             "is_empty":         is_empty,
#             "finish_reason":    finish,
#             "latency_ms":       round(latency_ms),
#             "reasoning_chars":  len(reasoning),
#             "difficulty":       row.get("difficulty", ""),
#             "category":         row.get("category", ""),
#             "reasoning_effort": label,
#             "model":            MODEL,
#         })
#         time.sleep(pause_s)
 
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)
 
#     n = len(sample)
#     answered = n - empty_count
#     raw_accuracy = correct_count / n * 100
#     answered_accuracy = (correct_count / answered * 100) if answered else 0.0
#     avg_latency = sum(r["latency_ms"] for r in results) / n
 
#     print(f"\nRaw accuracy (empties = wrong): {correct_count}/{n} = {raw_accuracy:.1f}%")
#     print(f"Answered-only accuracy: {correct_count}/{answered} = {answered_accuracy:.1f}%  "
#           f"(excludes {empty_count} empty)")
#     print(f"Avg latency: {avg_latency:.0f} ms")
#     print(f"Saved to: {output_path}")
 
#     return results
 
# # run test for all reasoning efforts
# def run_all_efforts(csv_path: str = TEST_DATASET_PATH, num_questions: int = 50, difficulty: str = None, category: str = None):
#     print("=" * 60)
#     print("  QWEN 3.6 dataset benchmark, 3 reasoning efforts")
#     print("=" * 60)
 
#     sample = load_sample(csv_path, num_questions, difficulty, category)
 
#     summary = {}
#     for effort in [None, "low", "medium"]:
#         label = "none" if effort is None else effort
#         output_path = f"{TEST_RESULTS_PATH}/test_results_qwen_openrouter_{label}.csv"
#         results = run_dataset_test_single(sample, effort, output_path)
 
#         n = len(results)
#         empty = sum(1 for r in results if r["is_empty"])
#         correct = sum(1 for r in results if r["is_correct"])
#         answered = n - empty
 
#         summary[label] = {
#             "n":            n,
#             "empty":        empty,
#             "correct":      correct,
#             "raw_acc":      correct / n * 100,
#             "answered_acc": (correct / answered * 100) if answered else 0.0,
#             "avg_latency":  sum(r["latency_ms"] for r in results) / n,
#         }
 
#     print("\n" + "=" * 60)
#     print("  FINAL COMPARISON")
#     print("=" * 60)
#     print(f"{'effort':<8} {'raw acc':>9} {'answered acc':>14} {'empty':>8} {'avg ms':>9}")
#     for label, s in summary.items():
#         print(f"{label:<8} {s['raw_acc']:>8.1f}% {s['answered_acc']:>13.1f}% "
#               f"{s['empty']:>5}/{s['n']:<3} {s['avg_latency']:>8.0f}")
#     print("=" * 60)
#     print(f"\nIndividual CSVs saved in {TEST_RESULTS_PATH}\n")
 
#     return summary

if __name__ == "__main__":
    if "--test" in sys.argv:
        # model = "qwen/qwen3.6-27b"
        model = "openai/gpt-5.4-mini"
        # model = "mistralai/mistral-medium-3-5"
        run_dataset_test(num_questions=50, model=model)
    else:
        # model = "qwen/qwen3.6-27b"
        # model = "openai/gpt-5.4-mini"
        model = "mistralai/mistral-medium-3-5"

        run_quiz(model)