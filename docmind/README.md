# DocMind: chat with your documents

Upload PDFs, DOCX or TXT files and ask questions. Answers cite the file and page they came from, and the app says so when the documents don't contain the answer.

**Pipeline:** parse (PyMuPDF) → sentence-aware chunks → BGE embeddings → Chroma + BM25 → Reciprocal Rank Fusion → cross-encoder re-rank → Claude answers with `[n]` citations.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY

# terminal 1 (backend)
uvicorn app.api.main:app --port 8000
# terminal 2 (frontend)
streamlit run app/ui/streamlit_app.py
```
Open http://localhost:8501. The first run downloads two small models (about 150 MB).

## Evaluate
1. Upload your documents in the app.
2. Fill `evaluation/testset.json` with 40-50 questions (file name + page of the answer).
3. `python -m evaluation.eval_retrieval`, then open the Evaluation tab.

## Docker
```bash
docker build -t docmind .
docker run -p 7860:7860 -e ANTHROPIC_API_KEY=your_key docmind
```

## Tests
`pytest tests/`
