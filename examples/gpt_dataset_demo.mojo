"""
GPT Dataset Demo
=================
Demonstrates LLMDataset + DataLoader with a sliding-window token
prediction task, using the built-in BasicTokenizer trained on the text.
"""

from bpe import BasicTokenizer
from tenmo.nlp import LLMDataset
from std.pathlib import Path


def main() raises:
    # Load the short story "The Verdict" by Edith Wharton
    var text_path = Path("the-verdict.txt")
    var text = text_path.read_text()

    # Train a tiny BPE tokenizer on the text itself
    var tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=512)  # small vocab for demo

    # Create dataset with sliding window
    var ds = LLMDataset(text, tokenizer, max_length=8, stride=4)

    # Create DataLoader from the dataset
    var dl = ds.into_loader(batch_size=2, shuffle=False, drop_last=True)

    # Iterate a few batches
    print("num_batches:", len(dl))
    print()

    var batch_count = 0
    for batch in dl:
        var x = batch.features
        var y = batch.labels
        print(
            "batch", batch_count,
            "| x shape:", x.shape(),
            "y shape:", y.shape(),
        )
        # Decode first sample of the batch
        var sample_tokens = List[Int](capacity=8)
        for j in range(8):
            sample_tokens.append(Int(x[0, j]))
        print("  input :", tokenizer.decode(sample_tokens))
        batch_count += 1
        if batch_count >= 3:
            break

    print("Done.")
