"""
Thin wrappers that let the graph call a model without knowing its runtime.

One rule governs this file: **a stub must never be mistakable for a real
inference.** Every stubbed result carries `stub=True`, the graph stamps STUB
into the trace, and the answer text says so. Screenshots taken during
development must not be able to masquerade as measured results in a report.

Real inference is wired per-runtime as artefacts land in models/weights/:
  onnx      -> onnxruntime CPU session (M1-M6)
  llamacpp  -> llama-server HTTP on 127.0.0.1:8080 (M7)
"""
from __future__ import annotations

import os
from typing import Any

from agent.registry import ModelSpec, Task

LLAMA_SERVER = os.environ.get("SATQUERY_LLAMA_URL", "http://127.0.0.1:8080")

# Warm ONNX sessions, created lazily and reused: session construction costs
# ~100ms and would otherwise dominate per-query latency.
_SESSIONS: dict[str, Any] = {}


def _session(spec: ModelSpec):
    if spec.id not in _SESSIONS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4          # 4 physical cores on this machine
        _SESSIONS[spec.id] = ort.InferenceSession(
            spec.weights_path, so, providers=["CPUExecutionProvider"])
    return _SESSIONS[spec.id]


def _stub(spec: ModelSpec, note: str = "") -> dict[str, Any]:
    reason = note or f"{spec.id} has no trained weights at {spec.weights}"
    return {
        "stub": True,
        "summary": f"[STUB — {reason}]",
        "confidence": None,
        "evidence": [],
    }


# --------------------------------------------------------------------------- #
def run_model(spec: ModelSpec, query: str, images: list[dict[str, Any]],
              params: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """Dispatch to the right runtime. Returns a dict with at least
    `summary`, `confidence`, `evidence`, `stub`."""
    if spec.runtime == "llamacpp":
        return _run_vlm(spec, query, images, params, **kw)
    if spec.runtime == "torch":
        if not spec.is_trained:
            return _stub(spec)
        try:
            from agent.torch_runtime import run as run_torch
            return run_torch(spec, query, images, params, **kw)
        except Exception as exc:
            return {"stub": True,
                    "summary": f"[ERROR {type(exc).__name__}: {exc}]",
                    "confidence": None, "evidence": []}
    if not spec.is_trained:
        return _stub(spec)
    try:
        return _run_onnx(spec, query, images, params)
    except Exception as exc:
        # A model that failed is a legitimate trace outcome; the graph records
        # the error and the run continues with the remaining specialists.
        return {"stub": True, "summary": f"[ERROR {type(exc).__name__}: {exc}]",
                "confidence": None, "evidence": []}


def _run_onnx(spec: ModelSpec, query: str, images: list[dict[str, Any]],
              params: dict[str, Any]) -> dict[str, Any]:
    if spec.id == "M2":
        return _run_m2_vqa(spec, query, images, params)
    # M1, M3, M4, M5a, M5b, M6 land here as each is trained and exported.
    return _stub(spec, f"{spec.id} weights present but no adapter implemented yet")


def _run_m2_vqa(spec: ModelSpec, query: str, images: list[dict[str, Any]],
                params: dict[str, Any]) -> dict[str, Any]:
    """Single-image VQA — the one model that is actually trained today.

    Preprocessing constants (mean, std, image_size, max_qlen) are read from the
    exported vocab file rather than duplicated here. Duplicating them would let
    the two copies drift apart, which degrades accuracy silently — the worst
    kind of bug, because the model still answers.
    """
    import json
    import numpy as np
    from PIL import Image

    vocab_path = spec.weights_path.replace(".onnx", "_vocab.json")
    if not os.path.exists(vocab_path):
        return _stub(spec, "vocab file missing next to the ONNX")
    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)

    word2id = vocab["word2id"]
    answers = vocab["answers"]
    max_qlen = vocab["max_qlen"]
    size = vocab["image_size"]

    # question -> ids (must match rsvqa.encode_question)
    toks = [t for t in query.lower().replace("?", " ").replace(",", " ").split() if t]
    ids = [word2id.get(t, word2id.get("<unk>", 1)) for t in toks][:max_qlen]
    mask = [1.0] * len(ids) + [0.0] * (max_qlen - len(ids))
    ids = ids + [word2id.get("<pad>", 0)] * (max_qlen - len(ids))

    path = images[0]["path"]
    if not os.path.exists(path):
        return _stub(spec, f"image not found: {path}")

    mean = np.array(vocab["mean"], dtype=np.float32)
    std = np.array(vocab["std"], dtype=np.float32)
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
    arr = arr.transpose(2, 0, 1)[None]                       # NCHW

    sess = _session(spec)
    feed = {"image": arr,
            "question_ids": np.array([ids], dtype=np.int64),
            "question_mask": np.array([mask], dtype=np.float32)}
    logits = sess.run(None, feed)[0][0]

    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top = int(probs.argmax())
    return {
        "stub": False,
        "answer": answers[top],
        "summary": f"answer={answers[top]!r}",
        "confidence": float(probs[top]),
        "evidence": [],
    }


def _run_vlm(spec: ModelSpec, query: str, images: list[dict[str, Any]],
             params: dict[str, Any], findings: list[str] | None = None,
             **kw: Any) -> dict[str, Any]:
    """M7 synthesis via llama-server's OpenAI-compatible endpoint.

    Falls back to a template that reports the specialist findings verbatim.
    The fallback is honest -- it states no VLM was available and repeats only
    what the specialists actually returned -- so the pipeline stays demoable
    before M7 is trained, without inventing prose.
    """
    findings = findings or []
    if not spec.is_trained or not _llama_up():
        if findings:
            body = "; ".join(findings)
            return {"stub": True,
                    "answer": f"[No VLM loaded — specialist findings verbatim] {body}",
                    "summary": "template fallback", "confidence": None, "evidence": []}
        return {"stub": True, "answer": "[No VLM loaded and no specialist output]",
                "summary": "template fallback", "confidence": None, "evidence": []}

    import json
    import urllib.request

    prompt = (
        "You are a remote-sensing analyst. Answer the user's question using "
        "ONLY the specialist model findings below. If they do not support an "
        "answer, say so. Do not speculate beyond the findings.\n\n"
        f"Question: {query}\n\nFindings:\n" +
        "\n".join(f"- {f}" for f in findings)
    )
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params.get("temperature", 0.2),
        "max_tokens": params.get("max_tokens", 256),
    }).encode()
    req = urllib.request.Request(
        f"{LLAMA_SERVER}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return {"stub": False,
            "answer": data["choices"][0]["message"]["content"].strip(),
            "summary": "synthesised", "confidence": None, "evidence": []}


def _llama_up() -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{LLAMA_SERVER}/health", timeout=1):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False
