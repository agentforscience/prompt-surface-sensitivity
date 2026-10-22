"""Greedy generation for every prompt in data/prompts.jsonl with one model (vLLM).

Usage: python src/generate.py --model Qwen/Qwen2.5-0.5B-Instruct --name qwen0.5b --gpu-gb 4.5

Outputs append to results/gen/<name>.jsonl (resumable: finished uids are skipped).
The chat template's default system prompt is kept, as a deployed model would use it.
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", "/data/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# In-process engine (faster startup); TP workers are forked from this process so they inherit
# the memory-snapshot patch below.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 3000


def gpu_total_gb():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.check_output(["nvidia-smi", "-i", dev, "--query-gpu=memory.total",
                                   "--format=csv,noheader,nounits"]).decode()
    return float(out.strip()) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu-gb", type=float, default=4.5)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--max-num-seqs", type=int, default=48)
    ap.add_argument("--batched-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--exps", default="cls,ifev,ext")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: items per (exp,task)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    exps = set(args.exps.split(","))
    prompts = [json.loads(l) for l in open(os.path.join(ROOT, "data", "prompts.jsonl"))]
    prompts = [p for p in prompts if p["exp"] in exps]
    if args.limit:
        ifev_keys = sorted({p["item"] for p in prompts if p["exp"] == "ifev"})[:args.limit]
        prompts = [p for p in prompts if (p["item"] in ifev_keys if p["exp"] == "ifev" else p["item"] < args.limit)]
    out_dir = os.path.join(ROOT, "results", "gen")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}{'_smoke' if args.limit else ''}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["uid"] for l in open(out_path)}
    todo = [p for p in prompts if p["uid"] not in done]
    print(f"[{args.name}] {len(todo)} to do ({len(done)} done)", flush=True)
    if not todo:
        return

    if args.tp > 1:
        # Each TP worker creates its CUDA context + NCCL buffers (~0.8 GB) *before* vLLM measures
        # free memory, so the vLLM budget must be ~0.4 GB below the slice the queue reserved.
        args.gpu_gb -= 0.4
    util = min(0.95, args.gpu_gb / gpu_total_gb())
    if args.tp == 1:
        # Shared-GPU race fix: grab the memory in *this* process as soon as it is free, and
        # release it inside vLLM's worker right before it measures free memory (in-process
        # engine), so other jobs cannot take it during vLLM's slow startup.
        import torch
        import vllm.v1.worker.gpu_worker as gw
        hold, deadline = [], time.time() + 180
        while not hold and time.time() < deadline:
            try:
                hold.append(torch.empty(int((args.gpu_gb + 0.3) * 1024 ** 3), dtype=torch.uint8, device="cuda"))
            except torch.OutOfMemoryError:
                time.sleep(0.3)
        if not hold:
            print("HOLD_TIMEOUT: could not reserve GPU memory", flush=True)
            return 3
        _snapshot = gw.MemorySnapshot

        def _release_then_snapshot(*a, **k):
            if hold:
                hold.clear()
                torch.cuda.empty_cache()
            return _snapshot(*a, **k)
        gw.MemorySnapshot = _release_then_snapshot
    else:
        # Tensor parallel: one holder process per GPU reserves the memory; each forked worker kills
        # the holder of its own GPU right before vLLM measures free memory.
        import signal
        import vllm.v1.worker.gpu_worker as gw
        phys = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        holders = {}
        for i, g in enumerate(phys):
            holders[i] = subprocess.Popen([sys.executable, os.path.join(ROOT, "src", "hold_gpu.py"),
                                           str(args.gpu_gb - 0.5), "180"], stdout=subprocess.PIPE, text=True,
                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=g))
        if any(h.stdout.readline().strip() != "READY" for h in holders.values()):
            for h in holders.values():
                h.kill()
            print("HOLD_TIMEOUT: could not reserve GPU memory", flush=True)
            return 3
        _snapshot = gw.MemorySnapshot

        def _release_then_snapshot(*a, **k):
            import torch
            h = holders.pop(torch.cuda.current_device(), None)
            if h is not None:
                os.kill(h.pid, signal.SIGKILL)
                t0 = time.time()
                while torch.cuda.mem_get_info()[0] < args.gpu_gb * 1024 ** 3 and time.time() - t0 < 10:
                    time.sleep(0.02)
                torch.cuda.empty_cache()
            return _snapshot(*a, **k)
        gw.MemorySnapshot = _release_then_snapshot
    llm = LLM(model=args.model, gpu_memory_utilization=util, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=max(args.batched_tokens, args.max_model_len),
              enforce_eager=True, enable_prefix_caching=True, seed=42, tensor_parallel_size=args.tp, trust_remote_code=False)
    if args.tp > 1:
        for h in holders.values():  # already killed by the workers; reap them
            h.kill()
            h.wait()
    tok = llm.get_tokenizer()

    # Long prompts last so cheap chunks checkpoint early.
    order = {"cls": 0, "ext": 1, "ifev": 2}
    todo.sort(key=lambda p: (order[p["exp"]], p["uid"]))
    t0 = time.time()
    with open(out_path, "a") as fout:
        for s in range(0, len(todo), CHUNK):
            batch = todo[s:s + CHUNK]
            convs = [[{"role": "user", "content": p["prompt"]}] for p in batch]
            sps = [SamplingParams(temperature=0.0, max_tokens=p["max_tokens"], seed=42) for p in batch]
            outs = llm.chat(convs, sps, use_tqdm=False)
            for p, o in zip(batch, outs):
                c = o.outputs[0]
                fout.write(json.dumps(dict(uid=p["uid"], text=c.text, n_tokens=len(c.token_ids),
                                           n_prompt_tokens=len(o.prompt_token_ids),
                                           finish=c.finish_reason), ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{args.name}] {s + len(batch)}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    # Record the exact chat-template rendering of one prompt for documentation
    with open(os.path.join(out_dir, f"{args.name}_template_example.txt"), "w") as f:
        f.write(tok.apply_chat_template([{"role": "user", "content": todo[0]["prompt"]}],
                                        tokenize=False, add_generation_prompt=True))
    print(f"[{args.name}] FINISHED in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
