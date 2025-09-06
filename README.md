# Fact Verification and Entity Linking with LLaMA2 and BERT

This project showcases a pipeline for **automated fact-checking and structured knowledge extraction** using both **Large Language Models (LLMs)** and **Named Entity Recognition (NER)**. It combines **LLaMA2 (7B)** for generative reasoning with **BERT (fine-tuned on CoNLL-2003)** for extracting named entities from both questions and LLM outputs.

The system performs four main tasks:
1. Generate raw answers using a local LLaMA2 model.
2. Extract named entities from both the input question and generated output.
3. Link entities to external knowledge bases like **Wikipedia** and **Wikidata**.
4. Validate the factual correctness of the answer based on evidence.

---

## Key Features

- **NER with BERT**: Detects named entities using `dbmdz/bert-large-cased-finetuned-conll03-english`.
- **Local LLM Integration**: Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) to query a locally hosted LLaMA 2 7B model.
- **Knowledge Base Linking**: Uses the **Wikipedia** and **Wikidata** APIs for entity disambiguation and verification.
- **Fact Checking**: Extracts yes/no answers or Wikipedia entity links and cross-checks them against external sources.

---

## 🛠 Tools & Technologies Used

| Skill / Tool              | Usage                                                                 |
|---------------------------|-----------------------------------------------------------------------|
| Python                    | Core programming language                                             |
| Hugging Face Transformers| BERT model loading and token classification pipeline                 |
| llama.cpp (`llama-cpp-python`) | Local inference with quantized LLaMA2 models                    |
| Regular Expressions       | Answer classification logic                                           |
| Requests API              | Interacting with Wikipedia and Wikidata                              |
| NLP / LLMs                | Entity extraction, text generation, reasoning                         |
| Named Entity Linking      | Context-aware entity disambiguation                                   |

---

## How It Works

### 🔹 Task 1: Generate Raw Output
- Input question is passed to a **local LLaMA2 model** using `llama-cpp-python`.
- The LLM returns a natural language response.

### 🔹 Task 2: Named Entity Recognition
- Entities are extracted from:
  - The **original question**
  - The **raw LLM output**
- Using a BERT-based model fine-tuned for NER (`dbmdz/bert-large-cased-finetuned-conll03-english`).

### 🔹 Task 3: Extract Answer
- If the question is binary (yes/no), regex is used to classify the LLM output.
- If the answer is fact-based, the system extracts entities and returns the most relevant **Wikipedia link**.

### 🔹 Task 4: Fact Validation
- The answer is validated by:
  - Fetching Wikipedia summaries for disambiguated entities.
  - Verifying if the answer and context words appear in the summary.

---
