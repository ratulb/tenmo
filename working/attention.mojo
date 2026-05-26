from bpe import GPT4Tokenizer, BasicTokenizer, RegexTokenizer
from tenmo.nlp import LLMDataset
from tenmo.dataloader import DataLoader
from tenmo import Embedding, Tensor, IntArray
from std.random import seed
from tenmo.common_utils import i, s

comptime dtype = DType.float32

def main() raises:
    seed()
    var tokenizer = GPT4Tokenizer()
    tokenizer.load_tiktoken("/home/tenmoomnet/bpe/data/o200k_base.tiktoken")
    #var tokenizer = BasicTokenizer()
    #var tokenizer = RegexTokenizer()
    #tokenizer.load_tiktoken("/home/tenmoomnet/bpe/data/o200k_base.tiktoken")

    var text: String
    with open("the-verdict.txt", "r") as f:
        text = f.read()
        print(len(text))
        #tokenizer.train(text, vocab_size=1000)
        #tokenizer.train(text, vocab_size=1000)
        print("Tokenizer length: ", len(tokenizer))
    var max_length = 4
    #var stride = 128
    var stride = max_length
    var dataset = LLMDataset(text, tokenizer, max_length=max_length, stride=stride)
    var loader = dataset.into_loader(batch_size=8, shuffle=True, drop_last=False)
    print("Num batches: ", len(loader))
    var data_iter = loader.__iter__()
    ref batch  = next(data_iter)
    var (inputs, _targets) = batch.features, batch.labels
    print("Batch size: ", batch.batch_size)
    _="""batch.features.print()
    batch.labels.print()
    var batch_num = 0
    for batch in loader:
        batch_num += 1
        if batch_num == len(loader):
            batch.features.print()
            batch.labels.print()
            print(len(batch.features), len(batch.labels))
    loader.reset()"""
    vocab_size = len(tokenizer)
    output_dim = 256
    embedding_layer = Embedding[dtype](num_embeddings=vocab_size, embedding_dim=output_dim)
    #embedding_layer.weight.print()
    #var input_ids = IntArray(2, 3, 5, 1)
    #embedding_layer(Tensor[DType.int64].d1([3])).print()
    print("inputs shape: ", inputs.shape())
    var token_embeddings = embedding_layer(inputs)
    print("token_embeddings shape: ", token_embeddings.shape())

    var context_length = max_length
    var pos_embedding_layer = Embedding[dtype](context_length, output_dim)
    var pos_embeddings = pos_embedding_layer(Tensor[DType.int64].arange(Int64(context_length)))
    print("pos_embeddings: ", pos_embeddings.shape())

    var input_embeddings = token_embeddings + pos_embeddings
    print("input_embeddings shape: ", input_embeddings.shape())

    var inputs_x = Tensor[dtype].d2(
        [[0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]]
    )
    inputs_x.print()
    var query = inputs_x[i(1), s()]
    query.print()

    for i, t in enumerate(inputs_x.slices()):
        print("i: ", i)
        t.print()
        #for ss in t.slices():
        for ss in t:
            print(ss[1])


def main_1() raises:
    var tokenizer = GPT4Tokenizer()
    tokenizer.load_tiktoken("/home/tenmoomnet/bpe/data/o200k_base.tiktoken")
    var enc_text: List[Int]
    with open("the-verdict.txt", "r") as f:
        raw_text = f.read()
        enc_text = tokenizer.encode(raw_text)
        print(len(enc_text))

    var enc_sample = enc_text[50:]
    #print(enc_sample)
    var context_size = 4
    _="""var x = enc_sample[:context_size]
    var y = enc_sample[1: context_size + 1]
    print("x:       ", x)
    print("y:             ", y)"""


    for i in range(1, context_size + 1):
        #var sliced = Slice(0, i)
        var context = enc_sample[0:i]
        var desired = enc_sample[i]
        print(tokenizer.decode(context), " -> ", tokenizer.decode([desired]))
