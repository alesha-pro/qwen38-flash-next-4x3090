# SPDX-License-Identifier: Apache-2.0
"""Frozen calibration/eval prompt sets for the FP8 KV A/B.

Two disjoint sets:
- CALIBRATION_PROMPTS: magnitude coverage for K/V scale collection
  (text, code, math, multilingual, long filler with embedded needles).
- EVAL_PROMPTS: held-out deterministic prompts for the output comparison.

Both sets are frozen; do not edit after the Phase 4 runs, or re-run all arms.
"""

CALIBRATION_PROMPTS = [
    "Explain the CAP theorem with two real-world examples.",
    "Write a Python function that merges two sorted iterators in O(n) time.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "Translate 'the quick brown fox jumps over the lazy dog' into French, German, and Japanese.",
    "Derive the quadratic formula step by step.",
    "Write a SQL query that finds the second-highest salary per department without window functions.",
    "Explain how TCP congestion control works, including slow start and AIMD.",
    "Напиши короткое эссе о влиянии книгопечатания на европейскую науку.",
    "Write a Rust struct for a concurrent ring buffer and explain the memory ordering choices.",
    "List ten differences between transformer and state-space sequence models.",
    "Describe the Raft consensus algorithm's leader election in detail.",
    "用中文解释什么是稀疏注意力机制，并举两个应用场景。",
    # long filler with embedded needles (positional/magnitude coverage)
    ("The following is a log of sensor readings. " + "Sensor 7 reports nominal. " * 400
     + "Sensor 7 reports ANOMALY code XJ-42 at timestamp 9917. "
     + "Sensor 7 reports nominal. " * 400
     + "What anomaly code did sensor 7 report and at what timestamp?"),
    ("Here is a story. " + "The cat sat on the mat. " * 500
     + "Suddenly, the cat revealed its name was Archimedes. "
     + "The cat sat on the mat. " * 300
     + "What was the cat's name?"),
    "Implement Dijkstra's algorithm in C++ with a binary heap.",
    "Explain the difference between FP8 E4M3 and E5M2 formats.",
    "Write a haiku about gradient descent.",
    "What are the security properties of TLS 1.3 vs TLS 1.2?",
    "Describe how a B-tree differs from a B+ tree and why databases prefer the latter.",
    "Explain the producer-consumer problem and solve it with semaphores in C.",
    "Give a step-by-step recipe for sourdough bread with baker's percentages.",
    "Explain Shannon's source coding theorem to a high-school student.",
    "Write a bash script that finds duplicate files by content hash.",
    "Discuss the trade-offs of row-major vs column-major memory layouts for GEMM.",
]

EVAL_PROMPTS = [
    "The capital of Australia is",
    "Write a one-line Python expression to reverse a string.",
    "What is 17 * 23? Answer with just the number.",
    "Name three primary colors.",
    "Complete: 'To be or not to be,'",
    "What is the chemical symbol for gold?",
    "In one word: what does HTTP stand for?",
    "What year did the Apollo 11 mission land on the Moon?",
    "Spell 'necessary' and use it in a sentence.",
    "What is the largest planet in the Solar System?",
    "Write a Python list comprehension that squares even numbers from 0 to 9.",
    "What is the boiling point of water at sea level in Celsius?",
    "Name the author of 'Pride and Prejudice'.",
    "What is the time complexity of binary search?",
    "Translate 'thank you' into Spanish.",
    "What is the square root of 144?",
]
