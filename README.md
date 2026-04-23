# Self-Refining GraphRAG

This repository contains the notebook implementation for **Self-Refining GraphRAG**, an iterative retrieval-augmented generation pipeline for multi-hop question answering. The code compares three settings on a 200-question slice of **HotpotQA distractor**:

- **Vanilla RAG**: dense semantic retrieval only
- **One-shot GraphRAG**: graph-based retrieval in a single pass
- **Self-Refining GraphRAG**: generate -> verify -> revise -> regenerate

The main idea is to retrieve an initial evidence set, generate a candidate answer, verify whether the answer is supported, and then revise the evidence graph using the verifier output before regenerating.

---

## File

- `Self-Refining_GraphRAG.ipynb`: main notebook containing data loading, retrieval pipelines, evaluation, ablations, plotting, and LaTeX table generation.

---

## Method Overview

### 1. Data preparation
The notebook loads the HotpotQA distractor development set from:

```text
data/hotpot_dev_distractor_v1.json
```

It then creates a **balanced 200-question slice** using random seed 42:

- 100 bridge questions
- 100 comparison questions

### 2. Vanilla RAG
Vanilla RAG retrieves the top-k paragraphs using sentence embeddings and cosine similarity between the question and each paragraph.

### 3. One-shot GraphRAG
For each question, the notebook builds a **per-question entity co-occurrence graph**:

- nodes = named entities + paragraph titles
- edges = co-occurrence within the same paragraph

Question entities are matched to graph nodes, expanded by 1 hop, and ranked using a hybrid score that combines:

- graph-based relevance
- semantic similarity

### 4. Self-Refining GraphRAG
The proposed method extends One-shot GraphRAG with an iterative loop:

1. Retrieve initial evidence
2. Generate an answer using the current evidence
3. Verify whether the answer is supported
4. Revise retrieval using:
   - weak entities from the verifier
   - entities from already retrieved paragraphs
   - semantic retrieval from the verifier's missing-information description
5. Append new paragraphs and regenerate
6. Stop when:
   - the answer is supported,
   - no new evidence is found, or
   - the max round budget is reached

---

## Requirements

### Python packages
Install the main dependencies:

```bash
pip install numpy pandas matplotlib tqdm requests networkx spacy sentence-transformers jupyter
```

Then install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

### Local LLM serving
The notebook calls a **local Ollama server** at:

```text
http://localhost:11434/api/generate
```

You must have Ollama running locally and at least one supported backbone available.

Example models used in the notebook:

- `qwen2.5:7b-instruct`
- `llama3.1:8b-instruct-q4_K_M`
- `gemma2:9b`

You can switch the active model in the notebook by editing the `MODEL` variable.

---

## Expected Directory Layout

```text
project/
├── data/
│   └── hotpot_dev_distractor_v1.json
├── figures/
├── tables/
├── Self-Refining_GraphRAG.ipynb
└── README.md
```

The notebook will also generate checkpoint files such as:

- `qwen_results_checkpoint.pkl`
- `llama_results_checkpoint.pkl`
- `gemma_results_checkpoint.pkl`

---

## How to Run

Open the notebook and run the cells in order.

### Notebook stages

1. **Data Engineering**
   - load HotpotQA
   - create the 200-question evaluation slice
   - inspect examples

2. **RAG Implementation**
   - define embedding-based retrieval
   - run Vanilla RAG

3. **One-shot GraphRAG**
   - build entity graphs
   - run graph-based retrieval
   - compare with Vanilla RAG

4. **Self-Refining GraphRAG**
   - define verifier prompt
   - define revision logic
   - run iterative refinement

5. **Ablations**
   - `R=1`
   - `R=2`
   - `R=3`
   - no-verifier variant

6. **Evaluation**
   - compute EM / F1 / recall
   - produce rescue / regression analysis
   - generate figures
   - export LaTeX tables

---

## Main Functions

### Data and preprocessing
- `get_corpus(question_item)`: converts a HotpotQA item into paragraph dictionaries
- `normalize(text)`: normalizes answer strings for evaluation
- `f1_score(prediction, ground_truth)`: token-level F1
- `exact_match_score(prediction, ground_truth)`: exact match
- `evaluate(results)`: computes row-level evaluation outputs

### Retrieval and generation
- `vanilla_retrieve(question, corpus, top_k=3)`: dense retrieval baseline
- `generate_answer(question, paragraphs)`: LLM answer generation from retrieved context
- `run_vanilla_rag(questions, top_k=3)`: runs Vanilla RAG on a list of questions

### GraphRAG
- `extract_entities(text)`: entity extraction with spaCy
- `build_graph(corpus)`: builds the entity co-occurrence graph
- `graph_retrieve(question, corpus, G, top_k=3, hops=1, alpha=0.5)`: graph-based retrieval
- `run_oneshot_graphrag(questions, top_k=3, hops=1)`: runs One-shot GraphRAG

### Self-refinement
- `verify_answer(question, paragraphs, answer)`: LLM-based verifier returning structured JSON
- `revise_and_retrieve(...)`: retrieves new paragraphs from verifier-guided revision
- `run_self_refining_graphrag(...)`: full iterative pipeline

---

## Outputs

### Metrics
The notebook computes:

- **EM**
- **F1**
- **Bridge F1**
- **Comparison F1**
- **Recall@3**
- **Average rounds used**
- **Rescue rate / regression rate**

### Figures
The notebook generates figures such as:

- `figures/f1_scatter.png`
- `figures/f1_vs_rounds.png`
- `figures/recall_by_method.png`

### LaTeX tables
The notebook exports LaTeX-ready tables into:

```text
tables/
```

including main results, ablations, and rescue/regression summaries.

---

## Notes and Assumptions

- The graph is built **per question**, not globally across the whole benchmark.
- Paragraph titles are injected as entities to reduce NER misses.
- If graph retrieval fails to return candidates, the code falls back to Vanilla RAG.
- The verifier uses constrained JSON output through Ollama.
- On JSON parse failure, the verifier defaults to `unsupported` so the loop continues safely.
- New paragraphs are **appended**, not replaced.

---

## Reproducing Multi-Backbone Results

To reproduce the comparison across Qwen, Llama, and Gemma:

1. Run the notebook once per backbone by changing `MODEL`
2. Save the corresponding checkpoint pickle
3. Re-run the evaluation/table cells after all checkpoint files are available

The notebook expects these files when building the combined comparison tables:

- `qwen_results_checkpoint.pkl`
- `llama_results_checkpoint.pkl`
- `gemma_results_checkpoint.pkl`

---

## Suggested Citation / Description

If you describe this code in a report, a concise summary is:

> Self-Refining GraphRAG is an iterative graph-based RAG pipeline for multi-hop QA that uses verifier feedback to revise the retrieved evidence before regeneration.

