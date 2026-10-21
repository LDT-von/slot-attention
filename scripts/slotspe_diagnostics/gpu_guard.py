"""Hard guard for GPU evaluation scripts.

WHY THIS EXISTS
---------------
The default `python` on PATH is torch 2.6.0+cu124, whose CUDA kernels cover
sm_50..sm_90 only.  This machine has an RTX 5090 (sm_120), so importing torch
and running a CUDA op succeeds at import time and then fails at the first kernel
launch with:

    RuntimeError: CUDA error: no kernel image is available for execution on the
    device

Any measurement taken with that interpreter is invalid.  This happened once in
this project and produced numbers that had to be discarded and retaken.

WHAT THIS DOES
--------------
`require_gpu_env()` verifies, before any work:
  1. torch.cuda.is_available()
  2. the installed torch CUDA arch list covers the live device's compute cap
  3. a trivial CUDA op actually executes

and raises `EnvironmentError` with actionable text otherwise.

USAGE
-----
    from _gpu_guard import require_gpu_env
    require_gpu_env()      # call once, at the top of main(), before loading data
"""

from __future__ import annotations

import sys

REQUIRED_ARCH_SM = 120  # RTX 5090 / Blackwell consumer
KNOWN_GOOD_ENVS = ("/home/ubuntu/.conda/envs/dct/bin/python",
                   "/home/ubuntu/.conda/envs/trisurv/bin/python")


def require_gpu_env(strict: bool = True) -> dict:
    """Verify this interpreter can actually run on the local GPU.

    Returns a small dict describing the verified environment.  Raises
    EnvironmentError if any check fails.
    """
    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "device_name": None,
        "compute_cap": None,
    }

    problems = []

    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is False")
    else:
        cap = torch.cuda.get_device_capability(0)
        info["compute_cap"] = cap
        info["device_name"] = torch.cuda.get_device_name(0)

        # torch >= 2.x exposes the compiled arch list; older builds may not.
        arch_list = info["arch_list"] or []
        major, minor = cap
        covered = any(
            a == f"sm_{major}{minor}"
            or (a.startswith("sm_") and a != "sm_" and int(a[3:]) >= major * 10 + minor)
            for a in arch_list
        )
        if arch_list and not covered:
            problems.append(
                f"device is sm_{major}{minor} but this torch build only ships "
                f"{arch_list}; kernels will fail at launch")

        # decisive check: actually run a kernel
        try:
            x = torch.randn(64, 64, device="cuda:0")
            float((x @ x).sum().item())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"trivial CUDA op failed: {exc}")

    if problems and strict:
        raise EnvironmentError(
            "Refusing to run: this interpreter cannot execute CUDA on this GPU.\n  - "
            + "\n  - ".join(problems)
            + "\n\nUse a working interpreter, e.g.:\n  "
            + "\n  ".join(KNOWN_GOOD_ENVS)
            + "\n(e.g.  /home/ubuntu/.conda/envs/dct/bin/python your_script.py)"
        )
    return info


if __name__ == "__main__":
    try:
        info = require_gpu_env()
        print("OK:", info)
    except EnvironmentError as e:
        print("FAIL:", e)
        raise SystemExit(1)