from tenmo.scheduler import StepLR, MultiStepLR, CosineAnnealingLR, WarmupCosineLR
from std.python import Python, PythonObject
from std.testing import assert_true, assert_equal, TestSuite


def _int_list(vals: Int, *rest: Int) -> List[Int]:
    var result = List[Int]()
    result.append(vals)
    for i in range(len(rest)):
        result.append(rest[i])
    return result^


def test_steplr_basic() raises:
    """StepLR: LR = base_lr * gamma^(epoch // step_size)."""
    var scheduler = StepLR[DType.float32](
        base_lr=Scalar[DType.float32](1.0),
        step_size=10,
        gamma=Scalar[DType.float32](0.1),
    )
    for _ in range(10):
        var lr = scheduler.step()
        assert_true(lr == Scalar[DType.float32](1.0))
    for _ in range(10):
        var lr = scheduler.step()
        assert_true(lr == Scalar[DType.float32](0.1))


def test_steplr_state_dict() raises:
    """StepLR: state_dict roundtrip preserves params + counter."""
    var sched = StepLR[DType.float32](
        base_lr=Scalar[DType.float32](0.01),
        step_size=5,
        gamma=Scalar[DType.float32](0.5),
    )
    for _ in range(12):
        _ = sched.step()
    var state = sched.state_dict()
    var restored = StepLR[DType.float32](
        base_lr=Scalar[DType.float32](0.01),
        step_size=5,
        gamma=Scalar[DType.float32](0.5),
    )
    restored.load_state_dict(state)
    # last_epoch starts at -1; 12 steps → last_epoch = 11
    assert_equal(restored.last_epoch, 11)
    assert_true(sched.get_last_lr() == restored.get_last_lr())


def test_multisteplr_basic() raises:
    """MultiStepLR: decays only at milestone epochs."""
    var milestones = _int_list(10, 20, 30)
    var sched = MultiStepLR[DType.float32](
        base_lr=Scalar[DType.float32](1.0),
        milestones=milestones,
        gamma=Scalar[DType.float32](0.1),
    )
    for _ in range(10):
        assert_true(sched.step() == Scalar[DType.float32](1.0))
    assert_true(sched.step() == Scalar[DType.float32](0.1))


def test_multisteplr_no_milestones() raises:
    """MultiStepLR: empty milestones = no decay."""
    var sched = MultiStepLR[DType.float32](
        base_lr=Scalar[DType.float32](0.5),
        milestones=List[Int](),
        gamma=Scalar[DType.float32](0.1),
    )
    for _ in range(50):
        assert_true(sched.step() == Scalar[DType.float32](0.5))


def test_multisteplr_state_dict() raises:
    """MultiStepLR: state_dict roundtrip preserves params + counter."""
    var milestones = _int_list(10, 20)
    var sched = MultiStepLR[DType.float32](
        base_lr=Scalar[DType.float32](0.01),
        milestones=milestones,
        gamma=Scalar[DType.float32](0.5),
    )
    for _ in range(15):
        _ = sched.step()
    var state = sched.state_dict()
    var restored_milestones = _int_list(10, 20)
    var restored = MultiStepLR[DType.float32](
        base_lr=Scalar[DType.float32](0.01),
        milestones=restored_milestones,
        gamma=Scalar[DType.float32](0.5),
    )
    restored.load_state_dict(state)
    # last_epoch starts at -1; 15 steps → last_epoch = 14
    assert_equal(restored.last_epoch, 14)
    assert_true(sched.get_last_lr() == restored.get_last_lr())


def test_cosineannealinglr_basic() raises:
    """CosineAnnealingLR: base_lr at epoch 0, eta_min at epoch T_max."""
    var sched = CosineAnnealingLR[DType.float32](
        base_lr=Scalar[DType.float32](1.0),
        T_max=10,
        eta_min=Scalar[DType.float32](0.0),
    )
    var lr1 = sched.step()
    assert_true(lr1 > Scalar[DType.float32](0.9))
    for _ in range(9):
        _ = sched.step()
    var lr10 = sched.step()
    assert_true(lr10 < Scalar[DType.float32](0.01))


def test_cosineannealinglr_state_dict() raises:
    """CosineAnnealingLR: state_dict roundtrip preserves params + counter."""
    var sched = CosineAnnealingLR[DType.float32](
        base_lr=Scalar[DType.float32](0.1),
        T_max=20,
        eta_min=Scalar[DType.float32](0.001),
    )
    for _ in range(7):
        _ = sched.step()
    var state = sched.state_dict()
    var restored = CosineAnnealingLR[DType.float32](
        base_lr=Scalar[DType.float32](0.1),
        T_max=20,
        eta_min=Scalar[DType.float32](0.001),
    )
    restored.load_state_dict(state)
    # last_epoch starts at -1; 7 steps → last_epoch = 6
    assert_equal(restored.last_epoch, 6)
    assert_true(sched.get_last_lr() == restored.get_last_lr())


def test_scheduler_integration() raises:
    """End-to-end: scheduler.step() feeds optimizer.set_lr()."""
    from tenmo.tensor import Tensor
    from tenmo.shared.shapes import Shape
    from tenmo.optim import SGD

    comptime dtype = DType.float32
    var w = Tensor[dtype].ones(Shape(2, 2), requires_grad=True)
    var params = List[Pointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(Pointer(to=w).unsafe_origin_cast[MutAnyOrigin]())
    var opt = SGD[dtype](params^, lr=0.1)
    var milestones = _int_list(3)
    var sched = MultiStepLR[dtype](
        base_lr=Scalar[dtype](0.1),
        milestones=milestones,
        gamma=Scalar[dtype](0.5),
    )
    for _ in range(5):
        w.seed_grad(Scalar[dtype](1.0))
        opt.step()
        opt.zero_grad()
        opt.set_lr(sched.step())
    # Tracing: after each opt.step():
    #   epoch 0: lr=0.1, w=1.0-0.1=0.9
    #   epoch 1: lr=0.1, w=0.9-0.1=0.8
    #   epoch 2: lr=0.1, w=0.8-0.1=0.7
    #   epoch 3: lr=0.1, w=0.7-0.1=0.6  (scheduler.step() after opt.step(), so LR change applies NEXT epoch)
    #   epoch 4: lr=0.05, w=0.6-0.05=0.55
    assert_true(w.all_close(Tensor[dtype].full(Shape(2, 2), 0.55)))


def test_warmup_cosine_boundaries() raises:
    """WarmupCosineLR.
    Lr(0)==0, lr(warmup)==max, lr(max)==min,
    midpoint of cosine ≈ (max+min)/2, floor past max_steps."""
    comptime dtype = DType.float32
    var sched = WarmupCosineLR[dtype](
        max_lr=Scalar[dtype](6e-4),
        min_lr=Scalar[dtype](6e-5),
        warmup_steps=100,
        max_steps=1100,
    )
    # Pure-query leg: lr_at must not advance the counter.
    assert_true(sched.lr_at(0) == Scalar[dtype](0))
    assert_true(sched.lr_at(100) == Scalar[dtype](6e-4))
    assert_true(sched.lr_at(1100) == Scalar[dtype](6e-5))
    assert_true(sched.lr_at(5000) == Scalar[dtype](6e-5))
    # Linear ramp: halfway through warmup is half the peak.
    assert_true(
        abs(Float64(sched.lr_at(50)) - 3e-4) < 1e-9,
    )
    # Cosine midpoint (step 600 of 100..1100): coeff = 0.5 exactly.
    assert_true(
        abs(Float64(sched.lr_at(600)) - 3.3e-4) < 1e-9,
    )
    assert_equal(sched.last_step, -1)


def test_warmup_cosine_step_and_resume() raises:
    """WarmupCosineLR.
    Step() walks 0→ramp→cosine→floor; state_dict
    round-trip preserves the counter so a resume never replays warmup."""
    comptime dtype = DType.float32
    var sched = WarmupCosineLR[dtype](
        max_lr=Scalar[dtype](0.1),
        min_lr=Scalar[dtype](0.01),
        warmup_steps=4,
        max_steps=8,
    )
    assert_true(sched.get_last_lr() == Scalar[dtype](0))
    assert_true(sched.step() == Scalar[dtype](0))  # t=0
    assert_true(
        abs(Float64(sched.step()) - 0.025) < 1e-7
    )  # t=1: 0.1*1/4
    _ = sched.step()  # t=2
    _ = sched.step()  # t=3
    assert_true(sched.step() == Scalar[dtype](0.1))  # t=4: peak
    for _ in range(3):
        var lr = sched.step()  # t=5,6,7: strictly decaying cosine
        assert_true(
            Float64(lr) < 0.1 and Float64(lr) > 0.01,
        )
    assert_true(sched.step() == Scalar[dtype](0.01))  # t=8: floor
    assert_true(sched.step() == Scalar[dtype](0.01))  # t=9: stays floored
    # Resume: counter (t=9) survives the round-trip; trajectory continues
    # at the floor instead of restarting warmup.
    var state = sched.state_dict()
    var restored = WarmupCosineLR[dtype](
        max_lr=Scalar[dtype](0.1),
        min_lr=Scalar[dtype](0.01),
        warmup_steps=4,
        max_steps=8,
    )
    restored.load_state_dict(state)
    assert_equal(restored.last_step, 9)
    assert_true(restored.step() == Scalar[dtype](0.01))


def test_warmup_cosine_no_warmup() raises:
    """WarmupCosineLR with warmup_steps=0 is pure cosine from max_lr."""
    comptime dtype = DType.float32
    var sched = WarmupCosineLR[dtype](
        max_lr=Scalar[dtype](0.2),
        min_lr=Scalar[dtype](0.0),
        warmup_steps=0,
        max_steps=10,
    )
    assert_true(sched.lr_at(1) < Scalar[dtype](0.2))
    assert_true(sched.lr_at(1) > Scalar[dtype](0.0))
    assert_true(sched.lr_at(10) == Scalar[dtype](0.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
