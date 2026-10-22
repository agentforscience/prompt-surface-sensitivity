"""Tiny GPU scheduler for a shared host: launch each generation job when some GPU
has enough free memory; requeue jobs that crash (e.g. startup OOM when another user
grabs memory). Time spent waiting never counts as a failed try.

Usage: python src/gpu_queue.py logs/jobs.txt
jobs.txt lines: name  hf_model_id  need  [extra generate.py args...]
  need = alternatives separated by '|', each "GB" (1 GPU) or "NxGB" (tensor parallel over N GPUs),
  e.g. "9|2x5.5" -> one GPU with 9 GB free, or two GPUs with 5.5 GB free each.
Status lines -> logs/queue_status.txt
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
MARGIN_GB = 0.4
MAX_FAILS = 25
STAGGER_S = 60


def free_gb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                   "--format=csv,noheader,nounits"]).decode()
    return {int(a): float(b) / 1024 for a, b in (l.split(",") for l in out.strip().splitlines())}


def status(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(ROOT, "logs", os.environ.get("QUEUE_STATUS", "queue_status.txt")), "a") as f:
        f.write(line + "\n")


def main():
    jobs = []
    for l in open(sys.argv[1]):
        if l.strip() and not l.startswith("#"):
            name, model, need, *extra = l.split()
            alts = []
            for a in need.split("|"):
                n, gb = a.split("x") if "x" in a else (1, a)
                alts.append((int(n), float(gb)))
            jobs.append(dict(name=name, model=model, alts=alts, extra=extra, fails=0))
    running, last_launch = {}, {}
    while jobs or running:
        for name, (proc, job, gpu) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            if rc == 0:
                status(f"DONE {name} gpu{gpu}")
            else:
                # Losing a memory race at startup is not the job's fault -> don't count it.
                tail = open(os.path.join(ROOT, "logs", f"gen_{name}.log"), errors="ignore").read()[-6000:]
                race = any(m in tail for m in ("on startup is less than", "CUDA error: out of memory",
                                               "No available memory for the cache", "HOLD_TIMEOUT",
                                               "CUDA out of memory"))
                job["fails"] += 0 if race else 1
                status(f"FAIL {name} gpu{gpu} rc={rc} (fail {job['fails']}{', startup race' if race else ''})")
                if job["fails"] < MAX_FAILS:
                    jobs.append(job)
                else:
                    status(f"GIVEUP {name}")
        free = free_gb()
        for job in list(jobs):
            ready = [g for g in free if time.time() - last_launch.get(g, 0) > STAGGER_S]
            pick = None
            for n, gb in job["alts"]:
                cands = sorted([g for g in ready if free[g] >= gb + MARGIN_GB], key=lambda g: -free[g])
                if len(cands) >= n:
                    pick = (cands[:n], gb)
                    break
            if not pick:
                continue
            gpus, gb = pick
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
            log = open(os.path.join(ROOT, "logs", f"gen_{job['name']}.log"), "a")
            cmd = [PY, os.path.join(ROOT, "src", "generate.py"), "--model", job["model"],
                   "--name", job["name"], "--gpu-gb", str(gb), "--tp", str(len(gpus))] + job["extra"]
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            gpu = ",".join(map(str, gpus))
            running[job["name"]] = (proc, job, gpu)
            jobs.remove(job)
            for g in gpus:
                last_launch[g] = time.time()
                free[g] -= gb
            status(f"LAUNCH {job['name']} gpu{gpu} need={len(gpus)}x{gb} free_before={free[gpus[0]] + gb:.1f}")
        time.sleep(5)
    status("ALL FINISHED")


if __name__ == "__main__":
    main()
