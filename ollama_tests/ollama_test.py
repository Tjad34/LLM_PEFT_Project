import subprocess
import time
import json

def ask_ollama(model, question):
    """Ask a question to an Ollama model and measure time"""
    start_time = time.time()
    
    result = subprocess.run([
        'ollama', 'run', model, question
    ], capture_output=True, text=True)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return {
        'model': model,
        'question': question,
        'answer': result.stdout,
        'time_taken': response_time
    }

# Test questions
questions = [
    "What is Python?",
    "Write a simple hello world in Python",
    "Explain quantum computing in simple terms"
]

# Test different models (make sure you have these downloaded)
models = ['hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF', 'gemma2:2b', 'mistral:7b']

# Run comparisons
results = []
for model in models:
    print(f"Testing {model}...")
    for question in questions:
        result = ask_ollama(model, question)
        results.append(result)
        print(f"Question: {question}")
        print(f"Time: {result['time_taken']:.2f}s")
        print("---")

# Save results to the results folder
import os
results_path = os.path.join('..', 'results', 'ollama_comparison.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {results_path}")