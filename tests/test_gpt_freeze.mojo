"""Freeze / unfreeze cascade for the GPT stack.

HOW TO RUN:
    pixi run mojo -I . tests/test_gpt_freeze.mojo
(from the tenmo repo root).
"""

from std.testing import assert_true, assert_false, TestSuite
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.shared.indexhelper import i, s
from tenmo.gpt import MLP, TransformerBlock, GPTEmbedding, GPTModel


def test_freeze_mlp() raises:
    comptime dtype = DType.float32
    var mlp = MLP[dtype](8)
    for k in range(len(mlp.parameters())):
        assert_true(mlp.parameters()[k][].requires_grad)
    mlp.freeze()
    for k in range(len(mlp.parameters())):
        assert_false(mlp.parameters()[k][].requires_grad)
    mlp.unfreeze()
    for k in range(len(mlp.parameters())):
        assert_true(mlp.parameters()[k][].requires_grad)


def test_freeze_block() raises:
    comptime dtype = DType.float32
    var block = TransformerBlock[dtype](8, 2)
    for k in range(len(block.parameters())):
        assert_true(block.parameters()[k][].requires_grad)
    block.freeze()
    for k in range(len(block.parameters())):
        assert_false(block.parameters()[k][].requires_grad)
    block.unfreeze()
    for k in range(len(block.parameters())):
        assert_true(block.parameters()[k][].requires_grad)


def test_freeze_model_wte_row() raises:
    comptime dtype = DType.float32
    var model = GPTModel[dtype, DType.int64](
        32, 8, 8, 2, 2, tie_weights=True
    )
    for k in range(len(model.parameters())):
        assert_true(model.parameters()[k][].requires_grad)
    model.freeze()
    for k in range(len(model.parameters())):
        assert_false(model.parameters()[k][].requires_grad)
    # The book's frozen check (§4.9): a word-table row wants no gradient.
    assert_false(model.wte_wpe.wte.weight.requires_grad)
    var row = model.wte_wpe.wte.weight[i(0), s()]
    assert_false(row.requires_grad)
    model.unfreeze()
    for k in range(len(model.parameters())):
        assert_true(model.parameters()[k][].requires_grad)
    assert_true(model.wte_wpe.wte.weight.requires_grad)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
