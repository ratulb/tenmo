from python import Python
from pathlib import Path, _dir_of_current_file
from os import abort, makedirs
from sys import argv


fn main() raises:
    tqdm = Python.import_module("tqdm")
    requests = Python.import_module("requests")
    if len(argv()) != 2:
        print(
            "You must enter the model name as a parameter, e.g.:"
            "mojo  download.mojo 124M"
        )
        abort()
    args = argv()
    model = args[1]
    path = _dir_of_current_file() / "models" / model
    if not path.exists():
        makedirs(path)
    subdir = "models/" + model
    filenames = InlineArray[StaticString, 7](
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    )

    try:
        for file_idx in range(len(filenames)):
            filename = filenames[file_idx]
            r = requests.get(
                "https://openaipublic.blob.core.windows.net/gpt-2/"
                + subdir
                + "/"
                + filename,
                stream=True,
            )
            file_path = Path(subdir) / filename
            with open(file_path, "w") as f:
                file_size = Int(r.headers["content-length"])
                pbar = tqdm.tqdm(
                    ncols=100,
                    desc="Fetching " + filename,
                    total=file_size,
                    unit_scale=True,
                )
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                chunk_size = 1000
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))  # update by actual chunk size
                pbar.close()
    except e:
        print(e)
