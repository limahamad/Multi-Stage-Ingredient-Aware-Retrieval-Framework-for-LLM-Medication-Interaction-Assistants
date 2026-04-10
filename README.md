# Term Project: Medication Safety Assistant

This project is a medication safety assistant built with a three-stage Retrieval-Augmented Generation (RAG) pipeline. It helps answer questions about medication interactions, ingredient overlap, brand-to-generic normalization, and general medication safety.

The system includes:

- A Streamlit web app for interactive use
- A CLI pipeline runner for debugging and experimentation
- A three-stage hybrid retriever using FAISS dense retrieval and BM25 lexical retrieval
- An evaluation script for test cases and RAGAS-based automated scoring

## Project Overview

The assistant processes medication questions in three stages:

1. Stage 1: Normalize medication mentions
   - Maps brand names, aliases, and abbreviations to canonical generic names
2. Stage 2: Retrieve ingredient information
   - Finds active ingredients, dosage forms, and therapeutic class
3. Stage 3: Retrieve interaction evidence and generate the final answer
   - Uses ingredient-level or class-level evidence to produce a structured safety response

The final output includes:

- Decision: `Safe`, `Caution`, `Not Safe`, or `Uncertain`
- Severity level
- Short answer
- Mechanism summary
- Safety advice
- Evidence summary

## Main Files

- `app.py`  
  Streamlit interface for the Medication Safety Assistant

- `app_three_stage.py`  
  Core three-stage RAG pipeline and Gemini-based reasoning

- `build_three_stage_indexes.py`  
  Builds FAISS and BM25 indexes for all three stages from the local JSON datasets

- `evaluation.py`  
  Runs the pipeline on evaluation test cases and computes automated RAGAS metrics

- `requirements.txt`  
  Python dependencies

- `testcases.json`  
  Evaluation inputs used by the scoring script

## Data and Index Structure

The project uses local JSON files under `data/`:

- `data/stage1_normalization_docs.json`
- `data/stage2_ingredient_docs.json`
- `data/stage3_interaction_docs.json`

Generated indexes are stored under `rag_index/`:

- `rag_index/stage1_normalization/`
- `rag_index/stage2_ingredients/`
- `rag_index/stage3_interactions/`

Each stage stores:

- `docs.index` for FAISS dense retrieval
- `chunks.pkl` for chunk metadata
- `bm25.pkl` for BM25 lexical retrieval

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Set environment variables

This project uses Gemini for generation and reasoning.

Required:

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

Optional:

```powershell
$env:GEMINI_MODEL="models/gemini-flash-lite-latest"
```

For RAGAS evaluation with Groq:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

## How to Run

### Run the Streamlit app

From the `Term_Project` folder:

```powershell
streamlit run app.py
```

The app includes:

- A dashboard
- A "My Medicines" page for pairwise medicine checking
- An "Ask a Question" page for free-form medication safety queries

### Run the CLI version

```powershell
python app_three_stage.py
```

This opens a terminal loop where you can type medication questions and inspect stage-by-stage outputs.

### Rebuild the indexes

Run this if the index files are missing or if the JSON datasets change:

```powershell
python build_three_stage_indexes.py
```

## Evaluation

To run the evaluation script:

```powershell
python evaluation.py
```

The script:

- Loads prompts from `testcases.json`
- Runs the full pipeline on each case
- Computes automated RAGAS metrics when supported keys and packages are available
- Saves results to CSV files

Output files:

- `evaluation_manual_ratings.csv`
- `evaluation_ragas_metrics.csv`

## Retrieval and Modeling Details

- Embedding model: `all-MiniLM-L6-v2`
- Dense retrieval: FAISS inner-product similarity
- Sparse retrieval: BM25
- Generator / reasoner: Gemini via `google-generativeai`

The retriever uses a hybrid score that combines normalized dense and BM25 scores to improve recall and ranking quality.

## Example Questions

- `Can I take Panadol with warfarin?`
- `Is it safe to use paracetamol while taking metoclopramide?`
- `Can I use Advil while I'm on warfarin?`
- `Is it okay to drink grapefruit juice while taking Zocor?`
- `Can I take Augmentin if I'm allergic to amoxicillin?`

## Notes and Limitations

- This system is a course project and demonstration tool.
- It depends on the quality and coverage of the local medication datasets.
- It should not be used as a replacement for professional medical advice.
- Patient-specific factors such as age, dose, timing, diagnoses, allergies, pregnancy, and kidney/liver status are not fully modeled.

## Disclaimer

This information does not replace advice from a pharmacist or physician. Always consult a healthcare professional before making medication decisions.
