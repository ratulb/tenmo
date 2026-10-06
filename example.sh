#!/usr/bin/bash
clear
# DATA-PREREQUISITE CONVENTION (read before running chained examples):
# examples/data/ is gitignored and ships EMPTY — every corpus, vocab and
# checkpoint below must be generated in order. The chains are:
#   TinyStories corpus:  python3 scripts/fetch_tinystories.py  ->  examples/data/tinystories_{train,val}.txt
#   TinyStories 8k vocab: ./example.sh tinystories_vocab      ->  examples/data/tinystories_8k.tiktoken
#   Stage-2 pilot:        ./example.sh tinystories_pilot      ->  examples/data/pilot_best.npy (~2 h)
#   Stage-2b pilot:       ./example.sh tinystories_pilot_8k   ->  examples/data/pilot_8k_best.npy
#   BERT vocab:           ./example.sh imdb_bert_vocab        ->  examples/data/imdb_8k.tiktoken
#   BERT pretrain:        ./example.sh imdb_bert_pretrain     ->  examples/data/imdb_bert_mlm_best.npy
# Skipping a step fails LOUDLY with "No such file or directory" — go back
# and run the producer first.
# Single source of truth for dispatchable targets (keep in sync with the case table below).
EXAMPLES="word2vec_cbow|imdb|imdb_v1|binary_mnist|mnist|mnist_adamw|mnist_native|mnist_mixed|mnist_mixed_dtypes|mnist_unified|mnist_gelu|mnist_gpu|mnist_gpu_prof|mnist_conv2d|mnist_conv2d_gpu|mnist_conv_tt_gpu|xor|spiral|reverse_sequence|sort_sequence|cifar_10|gpt_dataset_demo|gpt_overfit|gpt_epochs|gpt_generate|tinystories_smoke|tinystories_pilot|tinystories_pilot_8k|tinystories_generate|tinystories_vocab|imdb_bert_vocab|imdb_bert_pretrain|imdb_bert|mnist_py|mnist_sgd_py|mnist_adamw_py|mnist_dataloader_py"
# Check if an argument was provided
if [ $# -eq 0 ]; then
  echo "Error: No example specified"
  echo "Usage: $0 [$EXAMPLES]"
  exit 1
fi

DEBUG_MODE=""
if [ $# -ge 2 ] && [ "$2" = "d" ]; then
  DEBUG_MODE="-D LOGGING_LEVEL=debug"
fi

# Determine which test to run based on the argument
case $1 in
  word2vec_cbow)
    echo "Running word2vec CBOW training loop"
    pixi run mojo -I . $DEBUG_MODE examples/word2vec_cbow.mojo
    ;;

  imdb_v1)
    echo "Running IMDB sentiment training loop(v1)"
    pixi run mojo -I . $DEBUG_MODE examples/imdb_sentiment_v1.mojo
    ;;

  imdb)
    echo "Running IMDB sentiment training loop(v2)"
    pixi run mojo -I . $DEBUG_MODE examples/imdb_sentiment_v2.mojo
    ;;
  binary_mnist)
    echo "Running binary mnist training loop"
    pixi run mojo -I . $DEBUG_MODE examples/binary_mnist.mojo
    ;;
  mnist_gpu)
    echo "Running mnist gpu training loop"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_gpu.mojo
    ;;
  mnist)
    echo "Running mnist training loop"
    pixi run mojo -I . $DEBUG_MODE examples/mnist.mojo
    ;;
  mnist_adamw)
    echo "Running mnist training loop (AdamW)"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_adamw.mojo
    ;;
  mnist_native)
    echo "Running pure-Mojo mnist with the tensor-native DataLoader"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_native.mojo
    ;;
  mnist_unified)
    echo "Running unified mnist (auto CPU/GPU)"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_unified.mojo
    ;;
  mnist_gelu)
    echo "Running mnist training loop (GeLU)"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_gelu.mojo
    ;;
  xor)
    echo "Running xor training loop"
    pixi run mojo -I . $DEBUG_MODE examples/xor.mojo
    ;;
  spiral)
    echo "Running mojo spiral training loop"
    pixi run mojo -I . $DEBUG_MODE examples/spiral.mojo
    ;;
  reverse_sequence)
    echo "Running reverse_sequence transformer training loop"
    pixi run mojo -I . $DEBUG_MODE examples/reverse_sequence.mojo
    ;;
  sort_sequence)
    echo "Running sort_sequence transformer training loop"
    pixi run mojo -I . $DEBUG_MODE examples/sort_sequence.mojo
    ;;
  cifar_10)
    echo "Running mojo cifar_10 training loop"
    pixi run mojo -I . $DEBUG_MODE examples/cifar_10.mojo
    ;;
  mnist_conv2d)
    echo "Running mojo mnist_conv2d.mojo training loop"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_conv2d.mojo
    ;;
  mnist_conv2d_gpu)
    echo "Running mojo mnist_conv2d_gpu.mojo GPU training loop"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_conv2d_gpu.mojo
    ;;
  mnist_conv_tt_gpu)
    echo "Running mojo mnist_conv_tt_gpu.mojo TileTensor GPU training loop"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_conv_tt_gpu.mojo
    ;;
  mnist_gpu_prof)
    echo "Running mnist gpu profiled training loop"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_gpu_prof.mojo
    ;;
  mnist_mixed)
    echo "Running MixedSequential mnist training loop (all f32)"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_mixed.mojo
    ;;
  mnist_mixed_dtypes)
    echo "Running mixed-precision mnist training loop (f32 stem, f16 bottleneck)"
    pixi run mojo -I . $DEBUG_MODE examples/mnist_mixed_dtypes.mojo
    ;;
  gpt_dataset_demo)
    echo "Running GPT sliding-window dataset demo"
    pixi run mojo -I . $DEBUG_MODE examples/gpt_dataset_demo.mojo
    ;;
  gpt_overfit)
    echo "Running GPT single-batch overfit smoke"
    pixi run mojo -I . $DEBUG_MODE examples/gpt_overfit.mojo
    ;;
  gpt_epochs)
    echo "Running GPT multi-epoch training demo (shared epoch loops)"
    pixi run mojo -I . $DEBUG_MODE examples/gpt_epochs.mojo
    ;;
  tinystories_smoke)
    # PREREQ: python3 scripts/fetch_tinystories.py (corpus: tinystories_train.txt).
    echo "Running TinyStories Stage-1 dress rehearsal (Ep-18 pieces, small scale)"
    pixi run mojo -I . $DEBUG_MODE examples/tinystories_smoke.mojo
    ;;
  tinystories_pilot)
    # PREREQ: fetched corpus (see above). Trains ~2 h, saves examples/data/pilot_best.npy.
    echo "Running TinyStories Stage-2 pilot (shrink config, ~1M tokens)"
    pixi run mojo -I . $DEBUG_MODE examples/tinystories_pilot.mojo
    ;;
  tinystories_generate)
    # PREREQ: ./example.sh tinystories_pilot (needs examples/data/pilot_best.npy).
    echo "Running TinyStories generation demo (stage-2 checkpoint -> text)"
    pixi run mojo -I . $DEBUG_MODE examples/tinystories_generate.mojo
    ;;
  tinystories_vocab)
    # PREREQ: fetched corpus. Produces examples/data/tinystories_8k.tiktoken for pilot_8k.
    echo "Running TinyStories vocab spike (train/save/reload 8k BPE)"
    pixi run mojo -I . $DEBUG_MODE examples/tinystories_vocab.mojo
    ;;
  imdb_bert_vocab)
    # First link of the BERT chain. Produces examples/data/imdb_8k.tiktoken.
    echo "Running IMDB BERT vocab builder (train/save/reload 8k BPE + specials)"
    pixi run mojo -I . $DEBUG_MODE examples/imdb_bert_vocab.mojo
    ;;
  imdb_bert_pretrain)
    # PREREQ: ./example.sh imdb_bert_vocab. Saves examples/data/imdb_bert_mlm_{best,latest}.npy.
    echo "Running IMDB BERT MLM pretraining smoke (2Lx128, capped batches)"
    pixi run mojo -I . $DEBUG_MODE examples/imdb_bert_pretrain.mojo
    ;;
  imdb_bert)
    # PREREQ: ./example.sh imdb_bert_pretrain (needs examples/data/imdb_bert_mlm_best.npy).
    echo "Running IMDB BERT sentiment fine-tune (transfer + frozen-test report)"
    pixi run mojo -I . $DEBUG_MODE examples/imdb_bert.mojo
    ;;
  tinystories_pilot_8k)
    # PREREQ: fetched corpus + ./example.sh tinystories_vocab (needs tinystories_8k.tiktoken).
    # Saves examples/data/pilot_8k_best.npy.
    echo "Running TinyStories Stage-2b pilot (custom 8k vocab, ~1M tokens)"
    pixi run mojo -I . $DEBUG_MODE examples/tinystories_pilot_8k.mojo
    ;;
  gpt_generate)
    echo "Running GPT generate demo (sample + uncached loop)"
    pixi run mojo -I . $DEBUG_MODE examples/gpt_generate.mojo
    ;;
  mnist_py)
    echo "Running mnist training loop (Python bindings)"
    pixi run python examples/mnist.py
    ;;
  mnist_sgd_py)
    echo "Running mnist SGD training loop (Python bindings, per-batch)"
    pixi run python examples/mnist_sgd.py
    ;;
  mnist_adamw_py)
    echo "Running mnist AdamW training loop (Python bindings, per-batch)"
    pixi run python examples/mnist_adamw.py
    ;;
  mnist_dataloader_py)
    echo "Running mnist dataloader demo (Python bindings)"
    pixi run python examples/mnist_dataloader.py
    ;;

  *)
    echo "Error: Unknown example '$1'"
    echo "Available examples: ${EXAMPLES//|/, }"
    exit 1
    ;;
esac
