"""Package completed raw evidence, excluding the large patient tensor cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.research.slotspe_proof.acceptance import audit_return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Return ZIP must be outside the original results directory")
    if output.exists():
        raise FileExistsError("Retain existing return ZIP; use a new filename")
    audit = audit_return(root)
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
        and not p.relative_to(root).as_posix().startswith("prepared/cache/")]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "slotspe_server_return/" + path.relative_to(root).as_posix())
        archive.writestr("slotspe_server_return/RETURN_AUDIT.json", json.dumps(audit, ensure_ascii=False, indent=2))
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    print(json.dumps({"output": str(output), "sha256": digest.hexdigest(), "files": len(files),
        "checkpoints_included": True, "patient_tensor_cache_included": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
