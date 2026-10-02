from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def prepare(text_path: str, out_dir: str, vocab_size: int = 2048):
    text = Path(text_path).read_text(encoding="utf-8")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = int(len(text) * .9)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text[:split]], trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(out / "tokenizer.json"))
    for name, part in [("train", text[:split]), ("val", text[split:])]:
        (out / f"{name}.txt").write_text(part, encoding="utf-8")
        import numpy as np
        np.asarray(tok.encode(part).ids, dtype=np.int32).tofile(out / f"{name}.bin")
    return len(tok.get_vocab())


def load_tokenizer(path):
    return Tokenizer.from_file(str(path))
