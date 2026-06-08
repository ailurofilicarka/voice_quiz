import os
import time
import csv
import random
import sys
from dotenv import load_dotenv
from groq import Groq

# Load API key
load_dotenv()

# Default configuration
MODEL = "llama-3.3-70b-versatile"    # LLM
NUM_QUESTIONS = 5                    # default number of questions

TEST_DATASET_PATH = "archive/quiz_questions.csv"
TEST_RESULTS_PATH = "results/test_results.csv"

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
def generate_question(topic: str, question_number: int, previous_questions: list[str]) -> str:

    # Build a string listing previous questions so the LLM avoids repeating them
    if previous_questions:
        avoid = "\n".join(f"- {q}" for q in previous_questions)
        avoid_text = f"\n\nDo NOT repeat any of these questions that were already asked:\n{avoid}"
    else:
        avoid_text = ""

    response = get_client().chat.completions.create(
        model=MODEL,

        messages=[
            {
                # The system message defines the LLM's role and rules
                "role": "system",
                "content": (
                    "You are a quiz master. Your job is to generate clear, fair quiz questions. "
                    "Rules:\n"
                    "- Ask exactly ONE question per response\n"
                    "- The question must have a single, clear correct answer\n"
                    "- Do not include the answer in your response\n"
                    "- Do not add any explanation, numbering, or extra text — just the question itself\n"
                    "- Keep the question concise (one sentence)"
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
def evaluate_answer(question: str, user_answer: str) -> tuple[bool, str]:

    response = get_client().chat.completions.create(
        model=MODEL,

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
                    "- Keep your total response under 30 words\n\n"
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

        # temperature=0.0 for the consistent evaluation
        # LLM should always apply the same rules
        temperature=0.0,

        # short response needed
        max_tokens=60,
    )

    llm_response = response.choices[0].message.content.strip()
    is_correct = llm_response.upper().startswith("CORRECT")

    return is_correct, llm_response

# Terminal quiz mode
# runs only when u do: py grok_client.py
# main quiz loop
def run_quiz():

    print("\n" + "=" * 50)
    print("LLM: Groq /", MODEL)
    print("=" * 50)

    topic = input("\nEnter a topic (or press Enter for 'general knowledge': )").strip() \
            or "general knowledge"

    num_input = input(f"How many questions? (press Enter for {NUM_QUESTIONS}): ").strip()
    try:
        num_questions = int(num_input) if num_input else NUM_QUESTIONS
    except ValueError:
        num_questions = NUM_QUESTIONS

    print(f"\nStarting quiz: {num_questions} questions about '{topic}'")
    print("-" * 50)

    # quiz loop
    correct_count = 0
    previous_questions = []

    for i in range(1, num_questions + 1):

        # Generate question
        print(f"\nGenerating question {i} of {num_questions}...")
        question = generate_question(topic, i, previous_questions)
        previous_questions.append(question)
        print(f"\nQuestion {i}: {question}")

        # Wait for answer
        user_answer = input("Your answer: ").strip()

        # Handle empty answer
        if not user_answer:
            print("(no answer given — marked as wrong)")
            print("-" * 50)
            continue

        # Evaluate answer
        print("Evaluating...")
        is_correct, explanation = evaluate_answer(question, user_answer)

        if is_correct:
            correct_count += 1
            print(f"CORRECT {explanation}")
        else:
            print(f"WRONG {explanation}")

        print("-" * 50)

    # Final score
    print(f"\n{'=' * 50}")
    print(f"Quiz complete!")
    print(f"Your score: {correct_count} / {num_questions}")

    percentage = round((correct_count / num_questions) * 100)
    print(f"Percentage: {percentage}%")

    if percentage == 100:
        print("  Perfect score! Excellent work.")
    elif percentage >= 60:
        print("  Good job!")
    else:
        print("  Better luck next time!")

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

    # Save TSV
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