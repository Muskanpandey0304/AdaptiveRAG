"""Download the corpus PDFs listed in data/papers.txt from arXiv into data/raw/.

The papers are not committed (each keeps its own arXiv licence), so this fetches the exact versions the results were
measured on. Files already present are skipped, and requests are spaced out as arXiv asks of automated clients.

    python -m src.download_papers
"""

import time
import urllib.request
from pathlib import Path

PAPERS = Path("data/papers.txt")
RAW = Path("data/raw")
DELAY_S = 3.0


def paper_ids(path: Path = PAPERS) -> list[str]:
    """arXiv ids with version (e.g. '2306.05685v4'), skipping comments and blank lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split("\t")[0].strip() for line in lines if line.strip() and not line.startswith("#")]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    ids = paper_ids()
    for n, arxiv_id in enumerate(ids, 1):
        target = RAW / f"{arxiv_id}.pdf"
        if target.exists():
            continue
        request = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "AdaptiveRAG/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"{n}/{len(ids)} {arxiv_id}", flush=True)
        time.sleep(DELAY_S)
    print(f"{len(ids)} papers in {RAW}")


if __name__ == "__main__":
    main()
