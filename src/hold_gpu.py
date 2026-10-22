"""Reserve N GB on the (single) visible GPU as soon as it is free; print READY; hold until
killed (or MAX_HOLD seconds). Used by generate.py for tensor-parallel launches on shared GPUs."""
import os
import sys
import time

import torch

need = int(float(sys.argv[1]) * 1024 ** 3)
deadline = time.time() + float(sys.argv[2])
while time.time() < deadline:
    try:
        x = torch.empty(need, dtype=torch.uint8, device="cuda")
        print("READY", flush=True)
        parent, t0 = os.getppid(), time.time()
        # Safety: release when the launching process dies, and never hold longer than 15 min.
        while os.getppid() == parent and time.time() - t0 < 900:
            time.sleep(0.5)
        sys.exit(0)
    except torch.OutOfMemoryError:
        time.sleep(0.3)
print("HOLD_TIMEOUT", flush=True)
sys.exit(3)
