"""Qwen-2.5-7B pure decode loop — cross-partition LLM 및 baseline용.

CLI: python3 run_qwen_llm.py <label> <duration_s> [--rag]
- duration_s: continuous inference 시간
- --rag: FAISS retrieval을 앞에 붙임 (same-part memory-bound 워크로드용)

모델 경로: /hf/hub/models--Qwen--Qwen2.5-7B-Instruct/... 형식 (HF_HOME=/hf)
"""
import argparse, os, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--rag", action="store_true", help="prefix a mock FAISS retrieval step")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_name}", flush=True)

    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    prompts = [
        "Explain 5G physical layer channel estimation in 100 words.",
        "What is MMSE equalization and when is it used?",
        "Describe the LDPC decoder in 5G NR uplink.",
        "How does MIMO beamforming work in massive MIMO deployment?",
        "Summarize O-RAN architecture and its RIC components.",
    ]
    sp = SamplingParams(temperature=0.7, max_tokens=256, top_p=0.9)

    # optional: emulate RAG by prepending a fixed retrieval-like context
    if args.rag:
        rag_context = (
            "Retrieved context (top-3 chunks): "
            + "5G NR PHY layer uses OFDM with SCS 15/30/60/120 kHz. "
            + "PUSCH channel estimation uses DMRS pilots with LMMSE or LS. "
            + "Neural receivers can improve BLER vs classical MMSE under high SNR.\n\nQuestion: "
        )
        prompts = [rag_context + p for p in prompts]

    start = time.time()
    n_iters = 0
    while time.time() - start < args.duration_s:
        _ = llm.generate(prompts, sp)
        n_iters += 1
        if n_iters % 5 == 0:
            elapsed = time.time() - start
            print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters}  elapsed={elapsed:.1f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE  iters={n_iters}  label={args.label}", flush=True)

if __name__ == "__main__":
    main()
