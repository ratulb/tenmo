"""
MNIST Training with Mojo — GPU Profiling Build.
Times every phase and individual forward kernels to identify bottlenecks.

All forward ops use sync=False. Phase timers track where every ms goes.
First and last batch of epoch 1 run per-op breakdowns (sync=True on each
kernel) to compare warm-up/compilation vs steady-state.
"""

from tenmo.tensor import Tensor
from tenmo.optim import SGD
from tenmo.net import Linear, ReLU, Sequential
from tenmo.crossentropy import CrossEntropyLoss
from std.python import Python
from tenmo.numpy_interop import from_ndarray, numpy_dtype
from tenmo.dataloader import NumpyDataset, MNIST_MEAN, MNIST_STD
from tenmo.device import GPU
from std.time import perf_counter_ns
from std.sys import has_accelerator
from tenmo.mnemonics import mm
from tenmo.accuracy import Accuracy


def pct(x: Float64, total: Float64) -> Float64:
    return 100.0 * x / total if total > 0 else 0.0


def train_mnist() raises:
    comptime if not has_accelerator():
        raise Error("No GPU accelerator found. Use mnist.mojo for CPU training.")

    print("=" * 80)
    print("MNIST Training - GPU (Profiling Build)")
    print("=" * 80)
    print()

    # ========== Data Loading ==========
    print("Loading MNIST dataset...")
    var mnist = Python.import_module("mnist_datasets")
    var loader = mnist.MNISTLoader(folder="/tmp")

    var train_data = loader.load()
    var train_images = train_data[0]
    var train_labels = train_data[1]

    var test_data = loader.load(train=False)
    var test_images = test_data[0]
    var test_labels = test_data[1]

    print("  Train samples:", len(train_images))
    print("  Test samples:", len(test_images))
    print()

    # ========== Data Preparation ==========
    comptime FEATURE_DTYPE = DType.float32
    comptime LABEL_DTYPE = DType.int32

    train_images = train_images.astype(numpy_dtype(FEATURE_DTYPE))
    train_labels = train_labels.astype(numpy_dtype(LABEL_DTYPE))
    test_images = test_images.astype(numpy_dtype(FEATURE_DTYPE))
    test_labels = test_labels.astype(numpy_dtype(LABEL_DTYPE))

    var X_train = from_ndarray[FEATURE_DTYPE](train_images, copy=True)
    var y_train = from_ndarray[LABEL_DTYPE](train_labels, copy=True)
    var X_test = from_ndarray[FEATURE_DTYPE](test_images, copy=True)
    var y_test = from_ndarray[LABEL_DTYPE](test_labels, copy=True)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Data shapes:")
    print("  X_train:", X_train.shape())
    print("  y_train:", y_train.shape())
    print("  X_test:", X_test.shape())
    print("  y_test:", y_test.shape())
    print()

    # ========== DataLoaders ==========
    var train_batch_size = 64
    var test_batch_size = 64

    var train_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](
        X_train, y_train
    )
    var test_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](X_test, y_test)

    var train_loader = train_dataset.into_loader(
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
        normalize_mean=Float32(MNIST_MEAN),
        normalize_std=Float32(MNIST_STD),
    )
    var test_loader = test_dataset.into_loader(
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        normalize_mean=Float32(MNIST_MEAN),
        normalize_std=Float32(MNIST_STD),
    )

    print("DataLoaders:")
    print("  Train batches:", len(train_loader))
    print("  Test batches:", len(test_loader))
    print()

    # ========== Model Architecture ==========
    print("Building model...")
    var model = Sequential[FEATURE_DTYPE]()
    model.append(
        Linear[FEATURE_DTYPE](
            784, 128, init_method="he", bias_zero=True
        ).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](128, 32, init_method="he", bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](32, 10, init_method="he", bias_zero=True).into(),
    )
    print("  Architecture: 784 -> 128 -> 32 -> 10")
    print("  Total parameters:", model.num_parameters())
    print()

    # ========== Training Setup ==========
    var num_epochs = 15
    var learning_rate = Scalar[FEATURE_DTYPE](0.01)
    var momentum = Scalar[FEATURE_DTYPE](0.9)
    var weight_decay = Scalar[FEATURE_DTYPE](1e-4)
    var clip_norm = Scalar[FEATURE_DTYPE](1)
    var clip_value = Scalar[FEATURE_DTYPE](0.5)

    var criterion = CrossEntropyLoss[FEATURE_DTYPE]()

    # ========== Transfer Model to GPU ==========
    print("Transferring model parameters to GPU...")
    var gpu = GPU()
    model = model.to_gpu(gpu, stop_grad=True)
    print("  Model is now resident on GPU")
    print()

    var w1 = model.modules[0].layer[Linear[FEATURE_DTYPE, mm]].weight
    var b1 = model.modules[0].layer[Linear[FEATURE_DTYPE, mm]].bias.value()
    var w2 = model.modules[2].layer[Linear[FEATURE_DTYPE, mm]].weight
    var b2 = model.modules[2].layer[Linear[FEATURE_DTYPE, mm]].bias.value()
    var w3 = model.modules[4].layer[Linear[FEATURE_DTYPE, mm]].weight
    var b3 = model.modules[4].layer[Linear[FEATURE_DTYPE, mm]].bias.value()

    var optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        clip_value=clip_value,
    )

    print("Training configuration:")
    print("  Epochs:", num_epochs)
    print("  Batch size:", train_batch_size)
    print("  Learning rate:", learning_rate)
    print("  Momentum:", momentum)
    print()
    print("  Forward sync mode: sync=False (all queued async)")
    print("  Kernel profiling: first and last batch of epoch 1")

    print("=" * 80)
    var training_start = perf_counter_ns()
    var profile_epoch = 0
    var profile_first_done = False
    var profile_last_done = False

    # Per-op timing storage (first and last batch of profile_epoch)
    var f_mm1: Float64 = 0.0
    var f_b1: Float64 = 0.0
    var f_r1: Float64 = 0.0
    var f_mm2: Float64 = 0.0
    var f_b2: Float64 = 0.0
    var f_r2: Float64 = 0.0
    var f_mm3: Float64 = 0.0
    var f_b3: Float64 = 0.0
    var f_loss: Float64 = 0.0
    var l_mm1: Float64 = 0.0
    var l_b1: Float64 = 0.0
    var l_r1: Float64 = 0.0
    var l_mm2: Float64 = 0.0
    var l_b2: Float64 = 0.0
    var l_r2: Float64 = 0.0
    var l_mm3: Float64 = 0.0
    var l_b3: Float64 = 0.0
    var l_loss: Float64 = 0.0

    for epoch in range(num_epochs):
        var epoch_start = perf_counter_ns()

        if epoch == 10:
            optimizer.set_lr(optimizer.lr / 10)
        if epoch == 15:
            optimizer.set_lr(optimizer.lr / 10)

        model.train()
        criterion.train()
        var train_loss = Scalar[FEATURE_DTYPE](0.0)
        var train_correct = 0
        var train_total = 0

        var phase_data_ns: Int = 0
        var phase_forward_ns: Int = 0
        var phase_zerograd_ns: Int = 0
        var phase_backward_ns: Int = 0
        var phase_optstep_ns: Int = 0
        var phase_accuracy_ns: Int = 0
        var phase_count: Int = 0

        train_loader.reset()
        var batch_idx = 0
        var total_batches = len(train_loader)

        while train_loader.__has_next__():
            ref batch = train_loader.__next__()

            var t0 = perf_counter_ns()
            var features_gpu = batch.features.to_gpu(gpu, sync=False)
            var labels_gpu = batch.labels.to_gpu(gpu, sync=False)
            var t1 = perf_counter_ns()

            var pred: Tensor[FEATURE_DTYPE]
            var loss: Tensor[FEATURE_DTYPE]

            # Per-op profiling block: run with sync=True, record individual times
            var do_profile = epoch == profile_epoch and (
                (not profile_first_done and batch_idx == 0) or
                (not profile_last_done and batch_idx == total_batches - 1)
            )

            if do_profile:
                var t = perf_counter_ns()

                var h = features_gpu.matmul[track_grad=True](w1, sync=True)
                var t_mm1 = Float64(perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()

                h = h + b1
                var t_b1 = Float64(perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()

                h = h.relu[track_grad=True](sync=True)
                var t_r1 = Float64(perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()

                h = h.matmul[track_grad=True](w2, sync=True)
                var t_mm2 = Float64(perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()

                h = h + b2
                var t_b2 = Float64(perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()

                h = h.relu[track_grad=True](sync=True)
                var t_r2 = Float64(perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()

                h = h.matmul[track_grad=True](w3, sync=True)
                var t_mm3 = Float64(perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()

                pred = h + b3
                var t_b3 = Float64(perf_counter_ns() - t) / 1e6

                loss = criterion(pred, labels_gpu)
                var t_loss = Float64(perf_counter_ns() - t) / 1e6

                var k_sum = t_mm1 + t_b1 + t_r1 + t_mm2 + t_b2 + t_r2 + t_mm3 + t_b3 + t_loss

                if not profile_first_done:
                    profile_first_done = True
                    print("-- Forward kernel breakdown (first batch, sync=True) ---")
                else:
                    profile_last_done = True
                    print("-- Forward kernel breakdown (last batch, sync=True) ---")

                print("  matmul 784x128:       ", t_mm1, "ms")
                print("  bias_add 128:         ", t_b1, "ms")
                print("  relu 128:             ", t_r1, "ms")
                print("  matmul 128x32:        ", t_mm2, "ms")
                print("  bias_add 32:          ", t_b2, "ms")
                print("  relu 32:              ", t_r2, "ms")
                print("  matmul 32x10:         ", t_mm3, "ms")
                print("  bias_add 10:          ", t_b3, "ms")
                print("  crossentropy:         ", t_loss, "ms")
                print("  --------------------------------")
                print("  Sum:                  ", k_sum, "ms")
                print()

                # Store for later comparison
                if batch_idx == 0:
                    f_mm1 = t_mm1
                    f_b1 = t_b1
                    f_r1 = t_r1
                    f_mm2 = t_mm2
                    f_b2 = t_b2
                    f_r2 = t_r2
                    f_mm3 = t_mm3
                    f_b3 = t_b3
                    f_loss = t_loss
                elif batch_idx == total_batches - 1:
                    l_mm1 = t_mm1
                    l_b1 = t_b1
                    l_r1 = t_r1
                    l_mm2 = t_mm2
                    l_b2 = t_b2
                    l_r2 = t_r2
                    l_mm3 = t_mm3
                    l_b3 = t_b3
                    l_loss = t_loss
            else:
                pred = model(features_gpu, sync=False)
                loss = criterion(pred, labels_gpu, sync=False)

            var t2 = perf_counter_ns()

            optimizer.zero_grad()
            var t3 = perf_counter_ns()

            loss.backward()
            var t4 = perf_counter_ns()

            optimizer.step()
            var t5 = perf_counter_ns()

            train_loss += loss.item() * Float32(batch.batch_size)
            train_correct += Accuracy[FEATURE_DTYPE].compute(pred, labels_gpu, sync=False)
            train_total += batch.batch_size
            var t6 = perf_counter_ns()

            var exclude = epoch == profile_epoch and batch_idx == 0
            if not exclude:
                phase_data_ns += Int(t1 - t0)
                phase_forward_ns += Int(t2 - t1)
                phase_zerograd_ns += Int(t3 - t2)
                phase_backward_ns += Int(t4 - t3)
                phase_optstep_ns += Int(t5 - t4)
                phase_accuracy_ns += Int(t6 - t5)
                phase_count += 1

            batch_idx += 1

        # After profile epoch, print comparison
        if epoch == profile_epoch and profile_last_done:
            print("-- First vs last batch comparison ---")
            print("  matmul 784x128:     ", f_mm1, "ms ->", l_mm1, "ms")
            print("  bias_add 128:       ", f_b1, "ms ->", l_b1, "ms")
            print("  relu 128:           ", f_r1, "ms ->", l_r1, "ms")
            print("  matmul 128x32:      ", f_mm2, "ms ->", l_mm2, "ms")
            print("  bias_add 32:        ", f_b2, "ms ->", l_b2, "ms")
            print("  relu 32:            ", f_r2, "ms ->", l_r2, "ms")
            print("  matmul 32x10:       ", f_mm3, "ms ->", l_mm3, "ms")
            print("  bias_add 10:        ", f_b3, "ms ->", l_b3, "ms")
            print("  crossentropy:       ", f_loss, "ms ->", l_loss, "ms")
            print()

        model.eval()
        criterion.eval()
        var val_loss = Scalar[FEATURE_DTYPE](0.0)
        var val_correct = 0
        var val_total = 0

        test_loader.reset()
        while test_loader.__has_next__():
            ref batch = test_loader.__next__()

            var features_gpu = batch.features.to_gpu(gpu, sync=False)
            var labels_gpu = batch.labels.to_gpu(gpu, sync=False)

            var pred = model(features_gpu, sync=False)
            var loss = criterion(pred, labels_gpu, sync=False)

            val_loss += loss.item() * Float32(batch.batch_size)
            val_correct += Accuracy[FEATURE_DTYPE].compute(pred, labels_gpu, sync=False)
            val_total += batch.batch_size

        var epoch_time = Float64(perf_counter_ns() - epoch_start) / 1e9
        var avg_train_loss = train_loss / Float32(train_total)
        var train_acc = 100.0 * Float64(train_correct) / Float64(train_total)
        var avg_val_loss = val_loss / Float32(val_total)
        var val_acc = 100.0 * Float64(val_correct) / Float64(val_total)

        var s_data = Float64(phase_data_ns) / 1e9
        var s_forward = Float64(phase_forward_ns) / 1e9
        var s_zerograd = Float64(phase_zerograd_ns) / 1e9
        var s_backward = Float64(phase_backward_ns) / 1e9
        var s_optstep = Float64(phase_optstep_ns) / 1e9
        var s_accuracy = Float64(phase_accuracy_ns) / 1e9
        var s_total_phases = s_data + s_forward + s_zerograd + s_backward + s_optstep + s_accuracy
        var avg_ms = s_total_phases / Float64(phase_count) * 1000.0

        print("=" * 80)
        print("Epoch", epoch + 1, "/", num_epochs, "|", epoch_time, "s | total_phases:", s_total_phases, "s")
        print("Phase summary (", phase_count, " batches, sync=False forward)")
        print("  data_upload:     ", s_data, "s total,", s_data / Float64(phase_count) * 1000.0, "ms/batch,", pct(s_data, s_total_phases), "%")
        print("  forward:         ", s_forward, "s total,", s_forward / Float64(phase_count) * 1000.0, "ms/batch,", pct(s_forward, s_total_phases), "%")
        print("  zero_grad:       ", s_zerograd, "s total,", s_zerograd / Float64(phase_count) * 1000.0, "ms/batch,", pct(s_zerograd, s_total_phases), "%")
        print("  backward:        ", s_backward, "s total,", s_backward / Float64(phase_count) * 1000.0, "ms/batch,", pct(s_backward, s_total_phases), "%")
        print("  optimizer_step:  ", s_optstep, "s total,", s_optstep / Float64(phase_count) * 1000.0, "ms/batch,", pct(s_optstep, s_total_phases), "%")
        print("  accuracy_readback:", s_accuracy, "s total,", s_accuracy / Float64(phase_count) * 1000.0, "ms/batch,", pct(s_accuracy, s_total_phases), "%")
        print("  avg_batch:       ", avg_ms, "ms total")
        print("  Kernel launches (est.): forward=9, backward~20, opt~18 per batch")
        print("  Train Loss:", avg_train_loss, "| Train Acc:", train_acc, "% | Val Loss:", avg_val_loss, "| Val Acc:", val_acc, "%")

    var total_time = Float64(perf_counter_ns() - training_start) / 1e9
    print("=" * 80)
    print("Training completed in", total_time, "seconds")

    _ = model.to_cpu()
    print("  Model weights saved to CPU")
    print("=" * 80)


def main() raises:
    train_mnist()
