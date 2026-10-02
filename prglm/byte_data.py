import json
from pathlib import Path
import numpy as np


class ByteTokenizer:
    """Exact 256-symbol UTF-8 byte vocabulary shared by comparison models."""
    def encode(self, text):
        class Encoded:
            def __init__(self, ids): self.ids = ids
        return Encoded(list(text.encode("utf-8")))

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def get_vocab(self):
        return {str(i): i for i in range(256)}

    def save(self, path):
        Path(path).write_text(json.dumps({"type": "utf8-byte", "vocab_size": 256}))


def prepare_bytes(text_path, out_dir):
    raw = Path(text_path).read_bytes()
    split = int(len(raw) * .9)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.frombuffer(raw[:split], dtype=np.uint8).tofile(out / "train.bin")
    np.frombuffer(raw[split:], dtype=np.uint8).tofile(out / "val.bin")
    ByteTokenizer().save(out / "tokenizer.json")
    return len(raw[:split]), len(raw[split:])


def load_byte_data(out_dir):
    out = Path(out_dir)
    return np.fromfile(out / "train.bin", dtype=np.uint8), np.fromfile(out / "val.bin", dtype=np.uint8)
