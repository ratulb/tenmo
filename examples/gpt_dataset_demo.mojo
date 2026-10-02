"""
GPT Dataset Demo.
================
SlidingWindowDataset + WindowLoader over a
synthetic int64 token stream, with shift-by-one verification in
shuffled-enumerate and random-offset modes.

Against a real corpus the only extra step is tokenization at the call site
(mbpe): BPETokenizer().encode(text) gives List[Int], which crosses via
Tensor[DType.int64].from_list — the windowing core itself stays
tokenizer-free.
"""

from tenmo.dataloader import SlidingWindowDataset


def main() raises:
    # Synthetic stream of 64 IDs, each equal to its own position.
    var ids = List[Scalar[DType.int64]](capacity=64)
    for i in range(64):
        ids.append(Scalar[DType.int64](i))
    var ds = SlidingWindowDataset[DType.int64](ids^, seq_length=8, stride=2)
    print("windows:", len(ds))  # (64 - 8 + 2 - 1) // 2 = 28

    # Sequential peek: deterministic starts 0, 2, 4, 6.
    var peek = ds.into_loader(batch_size=4, shuffle=False)
    for batch in peek:
        print(
            "peek first starts:",
            Int(batch.features[0, 0]),
            Int(batch.features[1, 0]),
        )
        break

    # Enumerate mode, shuffled: every row shift-consistent.
    var loader = ds.into_loader(batch_size=4, shuffle=True, drop_last=False)
    print("enumerate batches:", len(loader))  # ceil(28 / 4) = 7
    var checked = 0
    for batch in loader:
        for s in range(batch.batch_size):
            for j in range(7):
                if batch.labels[s, j] != batch.features[s, j] + Int64(1):
                    raise Error("shift-by-one violation (enumerate)")
        checked += batch.batch_size
    print("enumerate rows checked:", checked)  # 28

    # Random-offset mode: fresh starts per batch, same invariant.
    var streamer = ds.into_loader(
        batch_size=4, shuffle=False, drop_last=True, random_offsets=True
    )
    print("random-offset batches:", len(streamer))  # (64 - 8) // 4 = 14
    var streamed = 0
    for batch in streamer:
        for s in range(batch.batch_size):
            var first = batch.features[s, 0]
            if first < Int64(0) or first > Int64(55):
                raise Error("random start out of bounds")
            for j in range(7):
                if batch.labels[s, j] != batch.features[s, j] + Int64(1):
                    raise Error("shift-by-one violation (random offsets)")
        streamed += batch.batch_size
    print("random-offset rows checked:", streamed)  # 56

    print("Done.")
