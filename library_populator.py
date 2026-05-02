#!/usr/bin/env python3
"""Continuously populate the Q&A library from Ollama servers. Runs independently."""
import os, json, time, random, urllib.request, urllib.error

LIBRARY_DIR = "library"
QUESTIONS_FILE = "programmingquestions.txt"

def generate_one():
    try:
        with open(QUESTIONS_FILE) as f:
            raw = f.read()
        questions = [q.strip() for q in raw.strip().split("\n") if q.strip() and q.strip() != "Advanced"]
    except:
        questions = ["Write a Python function."]
    question = random.choice(questions)

    try:
        with open("servers.json") as f:
            servers = json.load(f)
        workers = [s for s in servers if s.get("enabled", False) and s.get("role", "worker") == "worker"]
        remote = [s for s in workers if "localhost" not in s["url"] and "127.0.0.1" not in s["url"]]
        server = random.choice(remote) if remote else random.choice(workers) if workers else None
    except:
        server = None
    if server is None:
        print("  [POP] No workers available")
        return False
    model = server["model"]

    prompt = (f"Write a complete working Python solution for this problem. "
              f"Show only the code, no explanation.\n\nProblem: {question}")
    try:
        req = urllib.request.Request(f"{server['url']}/api/generate",
            data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            answer = json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        print(f"  [POP] Failed ({model}): {e}")
        return False

    if not answer:
        return False

    q_dir = os.path.join(LIBRARY_DIR, "questions")
    a_dir = os.path.join(LIBRARY_DIR, "answers")
    os.makedirs(q_dir, exist_ok=True)
    os.makedirs(a_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(q_dir) if f.endswith(".txt")])
    n = random.choice(existing).replace(".txt", "") if existing else str(len(os.listdir(q_dir)) + 1)

    with open(os.path.join(q_dir, f"{n}.txt"), "w") as f:
        f.write(question + "\n")
    with open(os.path.join(a_dir, f"{n}.txt"), "w") as f:
        f.write(answer + "\n")
    print(f"  [POP] Q&A #{n} from {model}: {question[:60]}...")

    # Seed with a best Q&A pair
    best_q_dir = os.path.join(LIBRARY_DIR, "questions", "bestquestions")
    best_a_dir = os.path.join(LIBRARY_DIR, "answers", "bestanswers")
    best_files = sorted([f for f in os.listdir(best_q_dir) if f.endswith(".txt")])
    if best_files:
        bf = random.choice(best_files)
        bq_path, ba_path = os.path.join(best_q_dir, bf), os.path.join(best_a_dir, bf)
        if os.path.exists(bq_path) and os.path.exists(ba_path):
            dst_n = random.choice(existing).replace(".txt", "") if existing else str(len(os.listdir(q_dir)) + 1)
            with open(bq_path) as f: best_q = f.read()
            with open(ba_path) as f: best_a = f.read()
            with open(os.path.join(q_dir, f"{dst_n}.txt"), "w") as f: f.write(best_q)
            with open(os.path.join(a_dir, f"{dst_n}.txt"), "w") as f: f.write(best_a)
            print(f"  [POP] Seeded #{dst_n}: {best_q[:60].strip()}...")

    return True

if __name__ == "__main__":
    print("[POP] Library populator started")
    while True:
        try:
            generate_one()
        except Exception as e:
            print(f"  [POP] Error: {e}")
        time.sleep(random.uniform(5, 15))
