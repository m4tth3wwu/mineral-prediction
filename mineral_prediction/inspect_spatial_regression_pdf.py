from pathlib import Path
import re
import sys

from pypdf import PdfReader


def main():
    path = Path(sys.argv[1])
    reader = PdfReader(path)
    terms = [
        "multi-ring",
        "multiple ring",
        "ring",
        "buffer",
        "annulus",
        "neighborhood",
        "spatial lag",
        "distance band",
        "多环",
        "缓冲",
        "邻域",
    ]
    print(f"path={path}")
    print(f"pages={len(reader.pages)}")
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        normalized = re.sub(r"\s+", " ", text)
        matches = [term for term in terms if term.lower() in normalized.lower()]
        if matches:
            print(f"\n--- page {page_number}; terms={matches} ---")
            lower = normalized.lower()
            positions = []
            for term in matches:
                start = 0
                while True:
                    position = lower.find(term.lower(), start)
                    if position < 0:
                        break
                    positions.append(position)
                    start = position + len(term)
            for position in sorted(set(positions))[:8]:
                left = max(0, position - 500)
                right = min(len(normalized), position + 900)
                print(normalized[left:right])
                print()


if __name__ == "__main__":
    main()
