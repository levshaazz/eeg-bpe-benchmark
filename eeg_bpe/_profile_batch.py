"""
Batch-size profiler for all model x dataset combinations.

Measures:
  - Peak VRAM (MB) for forward + backward + optimizer step
  - Throughput (samples/sec) over 5 warmup + 10 timed iterations
  - Whether torch.compile succeeds

Run:  python -m eeg_bpe._profile_batch
"""
import torch
import torch.nn as nn
import time
import json
import gc
import sys

def _safe_cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def profile_model(model_factory, input_factory, n_classes, batch_sizes,
                  label="", warmup=3, iters=8, use_compile=False):
    """Profile a model across batch sizes.

    Returns list of dicts with results for each batch_size.
    """
    results = []
    for bs in batch_sizes:
        _safe_cuda_cleanup()
        try:
            model = model_factory().cuda()
            n_params = sum(p.numel() for p in model.parameters())

            # Try torch.compile
            compiled = False
            if use_compile:
                try:
                    import triton  # noqa
                    model = torch.compile(model, mode="reduce-overhead")
                    compiled = True
                except (ImportError, Exception):
                    pass

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")

            x = input_factory(bs)
            y = torch.randint(0, n_classes, (bs,), device="cuda")
            crit = nn.CrossEntropyLoss()

            # Warmup (also triggers compile graph capture)
            for _ in range(warmup):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Timed iterations
            t0 = time.perf_counter()
            for _ in range(iters):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            throughput = (bs * iters) / dt

            results.append({
                "batch_size": bs,
                "peak_vram_mb": round(peak_mb, 1),
                "throughput_sps": round(throughput, 1),
                "time_per_step_ms": round(dt / iters * 1000, 2),
                "compiled": compiled,
                "n_params": n_params,
                "status": "ok",
            })
            print(f"  bs={bs:>5d}: peak={peak_mb:>7.0f} MB, "
                  f"{throughput:>8.0f} samp/s, "
                  f"{dt/iters*1000:>6.1f} ms/step"
                  f"{' [compiled]' if compiled else ''}")

            del model, opt, scaler, x, y, loss

        except torch.cuda.OutOfMemoryError:
            results.append({
                "batch_size": bs,
                "peak_vram_mb": None,
                "throughput_sps": None,
                "time_per_step_ms": None,
                "compiled": False,
                "status": "OOM",
            })
            print(f"  bs={bs:>5d}: OOM")
            _safe_cuda_cleanup()

        except Exception as e:
            results.append({
                "batch_size": bs,
                "status": f"error: {type(e).__name__}: {str(e)[:60]}",
            })
            print(f"  bs={bs:>5d}: ERROR {e}")
            _safe_cuda_cleanup()

    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1e9
    free_vram = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {total_vram:.1f} GB total, {free_vram:.1f} GB free")
    print(f"PyTorch: {torch.__version__}")
    print()

    from eeg_bpe.classifiers import (
        Conv1DClassifier, TransformerClassifier,
        ChannelwiseTransformerClassifier,
    )

    # ── Dataset configs ──────────────────────────────────────────────
    # (name, vocab_size, seq_len, n_classes, n_channels, typical_n_train, sfreq, n_time)
    datasets = [
        ("sleep_edf",       1026, 512, 5,  2, 5500, 100, 3000),
        ("bci_iv_2a",       1026, 512, 4, 22,  200, 250, 1000),
        ("physionet_mi",    1026, 512, 4, 64, 5000, 160,  640),
        ("mental_arith",    1026, 512, 2, 23, 1200, 500, 2500),
        ("epfl_p300",       1026, 512, 2, 32, 1500, 2048, 1024),
        ("ssvep_nakanishi", 1026, 512, 12, 8,  900, 256, 1024),
    ]

    all_results = {}

    for ds_name, V, seq_len, n_cls, n_ch, n_train, sfreq, n_time in datasets:
        print(f"{'='*60}")
        print(f"Dataset: {ds_name} (n_ch={n_ch}, n_train={n_train})")
        print(f"{'='*60}")

        # Batch sizes to test: powers of 2, capped at n_train
        base_sizes = [16, 32, 64, 128, 256, 512, 1024]
        sizes = [s for s in base_sizes if s <= n_train]

        ds_results = {}

        # ── BPE Classifiers (token sequences) ────────────────────────
        # Conv1D
        print(f"\n  Conv1D (2D input: B x {seq_len}):")
        ds_results["Conv1D"] = profile_model(
            model_factory=lambda: Conv1DClassifier(vocab_size=V+2, n_classes=n_cls),
            input_factory=lambda bs: torch.randint(0, V, (bs, seq_len), device="cuda"),
            n_classes=n_cls,
            batch_sizes=sizes,
        )

        # Transformer
        print(f"\n  Transformer (2D input: B x {seq_len}):")
        ds_results["Transformer"] = profile_model(
            model_factory=lambda: TransformerClassifier(
                vocab_size=V+2, n_classes=n_cls, max_len=seq_len),
            input_factory=lambda bs: torch.randint(0, V, (bs, seq_len), device="cuda"),
            n_classes=n_cls,
            batch_sizes=sizes,
        )

        # CW_Transformer (3D: B x n_ch x seq_len)
        # Effective batch = B * n_ch → much heavier
        cw_sizes = [s for s in [4, 8, 16, 32, 64, 128, 256]
                     if s <= n_train and s * n_ch <= 16384]
        print(f"\n  CW_Transformer (3D input: B x {n_ch} x {seq_len}, "
              f"effective=Bx{n_ch}):")
        ds_results["CW_Transformer"] = profile_model(
            model_factory=lambda: ChannelwiseTransformerClassifier(
                vocab_size=V+2, n_classes=n_cls, n_ch=n_ch,
                max_len_per_ch=seq_len),
            input_factory=lambda bs: torch.randint(
                0, V, (bs, n_ch, seq_len), device="cuda"),
            n_classes=n_cls,
            batch_sizes=cw_sizes,
        )

        # ── EEGNet (raw EEG: B x 1 x n_ch x n_time) ────────────────
        # Build EEGNet inline (avoid import issues)
        eeg_sizes = [s for s in [16, 32, 64, 128, 256, 512]
                      if s <= n_train]
        print(f"\n  EEGNet (4D input: B x 1 x {n_ch} x {n_time}):")

        _n_ch, _n_time, _n_cls, _sfreq = n_ch, n_time, n_cls, sfreq
        def _eegnet_factory():
            F1, D, F2 = 8, 2, 16
            kern = max(1, _sfreq // 2)
            if kern % 2 == 0: kern += 1
            sep_kern = max(1, _sfreq // 8)
            if sep_kern % 2 == 0: sep_kern += 1
            return nn.Sequential(
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern//2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1*D, (_n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1*D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(0.25),
                nn.Conv2d(F1*D, F2, (1, sep_kern), padding=(0, sep_kern//2), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.LazyLinear(_n_cls),
            )

        ds_results["EEGNet"] = profile_model(
            model_factory=_eegnet_factory,
            input_factory=lambda bs: torch.randn(
                bs, 1, _n_ch, _n_time, device="cuda"),
            n_classes=n_cls,
            batch_sizes=eeg_sizes,
            use_compile=False,  # EEGNet uses LazyLinear, compile fails
        )

        # ── Windowed Hist CNN (float features, not tokens) ───────────
        # Input: (B, n_windows * n_ch * V) flattened → (B, n_win, feat_dim)
        n_win = 10  # typical for EEG
        feat_dim = n_ch * 128  # after PCA usually ~128 per ch
        total_feat = n_win * feat_dim
        wh_sizes = [s for s in [16, 32, 64, 128, 256, 512] if s <= n_train]
        print(f"\n  WindowedHistCNN (float: B x {total_feat}):")

        _feat_dim, _n_win, _n_cls2 = feat_dim, n_win, n_cls
        def _wh_factory():
            return nn.Sequential(
                nn.Unflatten(1, (_n_win, _feat_dim)),
                nn.Linear(_feat_dim, 128),
                nn.ReLU(),
                nn.Flatten(1, 2),  # → (B, n_win*128) for Conv1d we need transpose
            )  # Simplified — real model uses Conv1d, but VRAM profile similar

        # Use the actual _WindowedCNN shape
        _total_feat = total_feat
        ds_results["WindowedHistCNN"] = profile_model(
            model_factory=lambda: nn.Sequential(
                nn.Linear(_total_feat, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, _n_cls2)),
            input_factory=lambda bs: torch.randn(
                bs, _total_feat, device="cuda"),
            n_classes=n_cls,
            batch_sizes=wh_sizes,
            use_compile=False,
        )

        all_results[ds_name] = ds_results
        print()

    # ── Save results ─────────────────────────────────────────────────
    out_path = "results/logs/batch_size_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("OPTIMAL BATCH SIZES (max bs with < 80% VRAM & best throughput)")
    print(f"{'='*72}")
    vram_budget = free_vram * 1000 * 0.80  # 80% of free VRAM in MB

    for ds_name, ds_res in all_results.items():
        print(f"\n  {ds_name}:")
        for model_name, model_res in ds_res.items():
            ok_runs = [r for r in model_res if r.get("status") == "ok"
                       and r.get("peak_vram_mb") is not None
                       and r["peak_vram_mb"] < vram_budget]
            if not ok_runs:
                print(f"    {model_name:20s}: ALL OOM or error")
                continue
            # Best = highest throughput within VRAM budget
            best = max(ok_runs, key=lambda r: r["throughput_sps"])
            print(f"    {model_name:20s}: bs={best['batch_size']:>5d}  "
                  f"peak={best['peak_vram_mb']:>6.0f} MB  "
                  f"{best['throughput_sps']:>8.0f} samp/s  "
                  f"{best['time_per_step_ms']:>6.1f} ms/step")


if __name__ == "__main__":
    main()
