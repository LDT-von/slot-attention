"""Read-only progress monitor for the two running SlotSPE queues.

It reports epoch progress, manifest status and the measured optimizer-step
rate for the anti-collapse queue and the diversity queue. It writes nothing
and starts no training.
"""

import glob
import json
import os
import time

ANTI = "local_evidence/anticollapse_blca_uni2h_folds0to4_seed3_run01"
DIV = "local_evidence/slotspe_diversity_scm_sro_v1"


def counts(root):
    out = {}
    for path in sorted(glob.glob(root + "/folds/*/arms/*/curve.json")):
        key = path.split("/folds/")[1].split("/arms/")[0] + "/" + path.split("/arms/")[1].split("/")[0]
        try:
            rows = json.load(open(path))
        except (OSError, ValueError):
            continue
        out[key] = (len(rows), max(row["epoch"] for row in rows))
    return out


def status(root):
    try:
        return json.load(open(root + "/queue_manifest.json")).get("status")
    except (OSError, ValueError):
        return "unreadable"


def main():
    base = {"anti": counts(ANTI), "div": counts(DIV), "t": time.time()}
    json.dump(base, open("/tmp/slotspe_monitor_base.json", "w"))
    print("monitor baseline recorded at", time.strftime("%H:%M:%S"))
    print("anti:", base["anti"])
    print("div :", base["div"])

    while True:
        time.sleep(1800)
        now = time.time()
        window = now - base["t"]
        anti, div = counts(ANTI), counts(DIV)
        print(f"\n--- {time.strftime('%H:%M:%S')}  window {window/60:.0f} min ---")
        print(f"anti manifest: {status(ANTI)}   div manifest: {status(DIV)}")
        for tag, new, old in (("anti", anti, base["anti"]), ("div", div, base["div"])):
            for key in sorted(new):
                steps, epoch = new[key]
                before = old.get(key, (steps, epoch))[0]
                rate = (steps - before) / window * 3600
                print(f"  {tag} {key}: epoch {epoch}/30, {steps} steps, {rate:.0f} steps/h")
        base = {"anti": anti, "div": div, "t": now}
        json.dump(base, open("/tmp/slotspe_monitor_base.json", "w"))


if __name__ == "__main__":
    main()
