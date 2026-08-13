import os
import time
import csv
import random
import sys
from dotenv import load_dotenv
from groq import Groq

from .hosts import build_evaluator_prompt, build_evaluator_user_message, DEFAULT_HOST
from .json_parse import parse_evaluator_response

# Load API key
load_dotenv()

# Default configuration
MODEL = "llama-3.3-70b-versatile"
NUM_QUESTIONS = 5                    # default number of questions

TEST_DATASET_PATH = "../../archive/quiz_questions.csv"
TEST_RESULTS_PATH = "../../results/test_results.csv"

# Client initialization
client = None       # will be created on first use

def get_client() -> Groq:
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in your .env file.")
        client = Groq(api_key=api_key)
    return client

# Asks the LLM to come up with one quiz question.
# Returns just the question text.
def generate_question(topic: str, question_number: int, previous_questions: list[str], model: str = MODEL) -> str:

    # Build a string listing previous questions so the LLM avoids repeating them
    if previous_questions:
        avoid = "\n".join(f"- {q}" for q in previous_questions)
        avoid_text = f"\n\nDo NOT repeat any of these questions that were already asked:\n{avoid}"
    else:
        avoid_text = ""

    response = get_client().chat.completions.create(
        model=model,

        messages=[
            {
                # The system message defines the LLM's role and rules
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
                "content": f"Generate question number {question_number} about: {topic}.{avoid_text}"
            }
        ],

        # temperature controls how creative/random the LLM is:
        # 0.0 = very predictable, always same answer
        # 1.0 = very creative, more varied output
        temperature=0.7,

        # max_tokens limits how long the response can be
        # this prevents the LLM from writing paragraphs
        max_tokens=100,

    )

    question = response.choices[0].message.content.strip()
    return question


# Asks LLM to judge whether the answer is correct
def evaluate_answer(question: str, user_answer: str, model: str = MODEL, host: str = DEFAULT_HOST,
                    question_number: int = None, total_questions: int = None, score: int = None, streak: int = None) -> tuple[bool, str]:

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_evaluator_prompt(host)},
            {"role": "user", "content": build_evaluator_user_message(question, user_answer, question_number, total_questions, score, streak)},
        ],

        # temperature=0.0 for the consistent evaluation
        # LLM should always apply the same rules
        temperature=0.6,

        # short response needed
        max_tokens=500,
    )

    llm_response_raw = response.choices[0].message.content

    return parse_evaluator_response(llm_response_raw)

# Terminal quiz mode: py grok_client.py
def run_quiz():

    print("\n" + "=" * 50)
    print("LLM: Groq /", MODEL)
    print("=" * 50)

    topic = input("\nEnter a topic (or press Enter for 'general knowledge'):").strip() \
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
        question = generate_question(topic, i, previous_questions)
        previous_questions.append(question)
        print(f"\nQuestion {i}: {question}")

        user_answer = input("Your answer: ").strip()

        if not user_answer:
            print("(no answer given — marked as wrong)")
            print("-" * 50)
            continue

        print("Evaluating...")
        is_correct, explanation = evaluate_answer(question, user_answer)

        if is_correct:
            correct_count += 1
        print(f"{explanation}")

        print("-" * 50)

    # Final score
    print(f"\n{'=' * 50}")
    print(f"Quiz complete!")
    print(f"Your score: {correct_count} / {num_questions}")

    percentage = round((correct_count / num_questions) * 100)
    print(f"Percentage: {percentage}%")
    print(f"{'=' * 50}\n")

# Gives the question to the LLM and asks it to answer it
def answer_question(question: str) -> tuple[str, float]:

    t_start = time.time()

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a trivia expert answering True/False questions. "
                    "Rules:\n"
                    "- Reply with exactly one word: either True or False\n"
                    "- No explanation, no punctuation, nothing else\n"
                    "- Examples: 'True' or 'False'"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.0,
        max_tokens=50,
    )

    answer = response.choices[0].message.content.strip()
    latency_ms = (time.time() - t_start) * 1000
    return answer, latency_ms

# Loads questions from trivia dataset, asks the LLM to answer
# each question, checks correctness against the known correct
# answer and saves the results.
def run_dataset_test(csv_path = TEST_DATASET_PATH, num_questions = 50, difficulty = None, category = None, output_path = TEST_RESULTS_PATH):

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

        llm_answer, latency_ms = answer_question(question)
        print(f" LLM answer: {llm_answer} ({latency_ms:.0f}ms)")

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
            "model":          MODEL,
        })

    # Summary
    accuracy = correct_count / len(sample) * 100
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print("=" * 50)
    print(f"  Model:     {MODEL}")
    print(f"  Questions: {len(sample)}")
    print(f"  Correct:   {correct_count} / {len(sample)}")
    print(f"  Accuracy:  {accuracy:.1f}%")
    print(f"  Avg. latency: {avg_latency:.0f}ms")
    print("=" * 50)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":

    if "--test" in sys.argv:
        run_dataset_test(num_questions=50, difficulty=None)
    else:
        run_quiz()