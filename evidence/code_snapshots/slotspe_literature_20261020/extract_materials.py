"""Read supplied PDFs/ZIPs without importing or running their code."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile
import fitz
import sys

sys.stdout.reconfigure(encoding="utf-8")

SOURCE = Path(r"C:\Users\栋栋\Desktop\slot attention")
OUT = Path(r"E:\DCT-Reg\docs\research\slotspe_literature_20261020")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "extracted_text").mkdir(exist_ok=True)
(OUT / "archive_text").mkdir(exist_ok=True)
manifest = {"stage": "SPE-08 literature identity and applicability audit", "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(SOURCE), "patient_data_read": False, "model_code_executed": False, "items": []}
for source in sorted(SOURCE.iterdir()):
    if not source.is_file():
        continue
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    item = {"file": str(source), "bytes": source.stat().st_size, "sha256": before}
    if source.suffix.lower() == ".pdf":
        with fitz.open(source) as doc:
            pages = [page.get_text(sort=True) for page in doc]
            item.update(pages=len(pages), metadata=doc.metadata)
            target = OUT / "extracted_text" / (source.stem + ".txt")
            target.write_text("\n\n".join(f"===== PDF PAGE {n+1} =====\n{text}" for n, text in enumerate(pages)), encoding="utf-8")
            item["text_path"] = str(target)
            item["first_page_excerpt"] = pages[0][:2200]
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            item["members"] = [{"name": f.filename, "bytes": f.file_size} for f in members if not f.is_dir()]
            texts = []
            for member in members:
                path = Path(member.filename)
                if member.file_size > 1_000_000 or member.is_dir():
                    continue
                if path.suffix.lower() not in {".md", ".py", ".yaml", ".yml", ".json", ".txt"}:
                    continue
                if any(part in {".git", "node_modules", "__pycache__"} for part in path.parts):
                    continue
                value = archive.read(member).decode("utf-8", errors="replace")
                texts.append({"archive_member": member.filename, "text": value})
            target = OUT / "archive_text" / (source.stem + ".json")
            target.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
            item["text_path"] = str(target)
    item["sha256_after_read"] = hashlib.sha256(source.read_bytes()).hexdigest()
    item["source_unchanged"] = item["sha256_after_read"] == before
    manifest["items"].append(item)
manifest["all_sources_unchanged"] = all(i["source_unchanged"] for i in manifest["items"])
(OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"items": len(manifest["items"]), "all_sources_unchanged": manifest["all_sources_unchanged"], "pdfs": [{"name": Path(i["file"]).name, "pages": i["pages"], "first_page_excerpt": i["first_page_excerpt"][:1400]} for i in manifest["items"] if "pages" in i], "archives": [{"name": Path(i["file"]).name, "members": len(i["members"]), "readmes": [m["name"] for m in i["members"] if "readme" in m["name"].lower()]} for i in manifest["items"] if "members" in i]}, ensure_ascii=False, indent=2))
