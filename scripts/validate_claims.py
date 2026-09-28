from pathlib import Path


root = Path(__file__).resolve().parents[1]
readme = (root / "README.md").read_text(encoding="utf-8")
assert "long1b_best" not in readme
assert "did **not**\noutperform the baseline" in readme
assert "not\na fair general optimizer ranking" in readme
print("pretraining public-claim validation: PASS")

