# Evaluation

Benchmarks and metrics for the CyberPhi fine-tuned model.

## Evaluation approaches

### 1. Held-out CTF challenges
A subset of CTF writeups is withheld from training.
The model is given the challenge description and scored on:
- Whether its answer contains the correct flag pattern
- Correctness of the identified vulnerability class
- Quality of the exploitation reasoning

### 2. SecurityEval benchmark
[SecurityEval](https://github.com/s3c2/SecurityEval) contains 130 Python
security coding tasks. The model generates code and a static analyser
(Bandit / Semgrep) checks for the expected vulnerability pattern.

### 3. CWE-knowledge Q&A
A manually curated set of 200 factual Q&A pairs covering common CWEs.
Evaluated with ROUGE-L and BERTScore against reference answers.

## Running evaluations

```bash
# Full benchmark suite
python evaluation/benchmarks/run_benchmarks.py --model cyberphi

# Specific benchmark
python evaluation/benchmarks/run_benchmarks.py --benchmark securityeval
python evaluation/benchmarks/run_benchmarks.py --benchmark ctf-held-out

# Metrics only (from a saved results JSON)
python evaluation/metrics.py --results outputs/eval_results.json
```

## Interpreting results

| Metric | Target |
|--------|--------|
| CTF flag extraction accuracy | > 40% |
| SecurityEval vulnerability detection | > 60% |
| CWE Q&A ROUGE-L | > 0.45 |
| <think> block present | > 90% of responses |
