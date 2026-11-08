"""Score the JSONL produced by rag_filter_pipeline.py with RAGAS (local Ollama judge) -> scores.csv.
Usage: python evaluate_ragas.py runs/test.jsonl scores.csv [--judge qwen2.5:7b]
The judge MUST differ from the generator (llama3.1:8b). Resumable: finished (qid, condition) rows are skipped.
Failed metric calls are written as 'nan' (never replaced by 0 or dropped silently); paired_analysis.py reports how many.
"""
import argparse, csv, json, os, math, sys, types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    m = types.ModuleType("langchain_community.chat_models.vertexai")
    class DummyChatVertexAI: pass
    m.ChatVertexAI = DummyChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = m

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerCorrectness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

COLS = ["qid", "condition", "faithfulness", "answer_correctness", "context_precision", "context_recall"]
KEYMAP = {"faithfulness": "faithfulness", "answer_correctness": "answer_correctness",
          "llm_context_precision_with_reference": "context_precision", "context_recall": "context_recall"}

ap = argparse.ArgumentParser()
ap.add_argument("jsonl"); ap.add_argument("out")
ap.add_argument("--judge", default="qwen2.5:7b")
ap.add_argument("--timeout", type=int, default=600)
a = ap.parse_args()

judge = LangchainLLMWrapper(ChatOllama(model=a.judge, temperature=0, num_ctx=8192, seed=42))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))
metrics = [Faithfulness(), AnswerCorrectness(), LLMContextPrecisionWithReference(), LLMContextRecall()]
cfg = RunConfig(timeout=a.timeout, max_workers=1, max_retries=3, seed=42)

rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8")]
rows = [r for r in rows if "condition" in r]
done = set()
if os.path.exists(a.out):
    done = {(r["qid"], r["condition"]) for r in csv.DictReader(open(a.out, encoding="utf-8"))}
new = not os.path.exists(a.out)
with open(a.out, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    if new:
        w.writeheader()
    for i, r in enumerate(rows, 1):
        if (r["qid"], r["condition"]) in done:
            continue
        ds = EvaluationDataset(samples=[SingleTurnSample(
            user_input=r["question"], response=r["answer"],
            retrieved_contexts=r["contexts"], reference=r["reference"])])
        res = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb, run_config=cfg,
                       raise_exceptions=False, show_progress=False)
        d = res.to_pandas().iloc[0].to_dict()
        out = {"qid": r["qid"], "condition": r["condition"]}
        for k, col in KEYMAP.items():
            v = d.get(k, float("nan"))
            out[col] = "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)
        w.writerow(out); f.flush()
        print(f"[{i}/{len(rows)}] {out}")
