# Waveform or Rhythm: BPE Tokenisation Benchmark for EEG

Reproducibility package for the paper:

> **Насыбуллин А. А., Корнаев А. В.** Форма волны или ритм: бенчмарк BPE-токенизации для сигналов ЭЭГ // *Известия Юго-Западного государственного университета. Серия: Информатика и вычислительная техника.* 2026. [in press]
>
> *English title:* Nasybullin A. A., Kornaev A. V. Waveform or rhythm: BPE tokenisation benchmark for EEG. *Proceedings of the Southwest State University. Computer Science, Computer Engineering and Control.* 2026.

**Paper PDF**: [`writing/paper_journal.pdf`](writing/paper_journal.pdf) (27 pages)
**LaTeX source**: [`writing/paper_journal.tex`](writing/paper_journal.tex)

## What this paper does

We test **when BPE (Byte Pair Encoding) tokenisation adds value for EEG classification** — a subword compression method borrowed from NLP — across six public datasets and five brain–computer-interface paradigm types. The central finding:

> **BPE tokenisation is effective when a paradigm encodes discriminative information in the waveform (amplitude domain); it fails when the information lies in spectral power modulation (frequency domain).**

## Headline results

| Paradigm | Best BPE variant | κ | Best baseline | Gap |
|---|---|---|---|---|
| Sleep-EDF (amplitude) | BPE_WindowedSeq_CNN 85.1% | 0.688 | EEGNet 85.3% | −0.2 pp |
| P300 (amplitude) | BPE_Windowed_LogReg 72.6% | 0.258 | EEGNet 88.0% | −15.4 pp |
| Mental Arithmetic (mixed) | BPE_Windowed 59.5% | 0.189 | PSD 68.2% | −7.7 pp |
| BCI-IV-2a (frequency) | BPE_WindowedSeq_CNN 30.6% | 0.074 | EEGNet 45.1% | −14.5 pp |
| PhysioNet-MI (frequency) | BPE_Windowed_LogReg 42.7% | **0.236** | EEGNet 58.2% | −15.5 pp |
| SSVEP (frequency) | all BPE ≈ chance | ≈0 | SSVEP-FFT 65.4% | −56.7 pp |

- BPE detected a **K-complex morphology** in Sleep-EDF from raw tokens without prior neurophysiological knowledge (Fig. 10 in paper).
- Token frequencies follow a **Zipf power law** (α ≈ −1.83, R² ≈ 0.895) — same long-tail structure as natural language.
- Ablation A7: BPE merge order adds **+18.0 pp** over raw amplitude bins and **+5.9 pp** over random merges on Sleep-EDF, confirming merges carry genuine information.

Full benchmark and ablations: see Table 3 and Ablation Studies section of the paper.

## Repository layout

```
.
├── LICENSE                     MIT
├── pyproject.toml              package metadata (eeg-bpe v1.0.0)
├── requirements.txt            pip dependencies
├── CITATION.cff                citation info
├── eeg_bpe/                    experiment code (26 .py files)
│   ├── bpe_engine.py           BPE train/apply (doubly-linked list + max-heap, O(n log n))
│   ├── quantization.py         adaptive / uniform / μ-law
│   ├── data_loading.py         6 dataset loaders + disk/memory cache
│   ├── classifiers.py          Conv1D, Transformer, ChannelwiseTransformer, WindHistCNN
│   ├── baselines.py            EEGNet, CSP+LDA, PSD, VQ, SSVEP-FFT, Patching
│   ├── exp2_downstream.py      main classification experiment
│   ├── ablations.py            A1-A15 ablation runners
│   ├── config.py               constants: seeds, batch sizes, defaults
│   └── run_all.py              master runner
├── scripts/                    result-processing scripts
├── results/logs/               experiment result CSVs + JSONs (134 files)
├── data_download/              dataset download scripts (raw data NOT bundled)
├── writing/                    paper source
│   ├── paper_journal.tex       LaTeX manuscript
│   ├── paper_journal.pdf       compiled PDF (27 pp)
│   └── figures/                figure sources (PDF + PNG)
└── CONTRIBUTING.md
```

## Reproducing the experiments

### Prerequisites

- Python 3.11 (also tested on 3.10, 3.12)
- MNE-Python, NumPy, SciPy, pandas, scikit-learn, PyTorch, statsmodels
- One CUDA GPU recommended (all experiments were run on RTX 5070 Ti)

```bash
pip install -e .              # installs the eeg_bpe package
# or:
pip install -r requirements.txt
```

### Download datasets

```bash
python -m data_download.download_bci_iv_2a
python -m data_download.download_physionet_mi
python -m data_download.download_sleep_edf
python -m data_download.download_epfl_p300
python -m data_download.download_mental_arithmetic
python -m data_download.download_ssvep_nakanishi
# or all at once:
python -m data_download.download_all
```

Datasets are ~12 GB in total. Sleep-EDF and PhysioNet-MI are the biggest.

### Quick smoke test (3 subjects × V=1024, single seed)

```bash
python -m eeg_bpe.run_all --quick
```

Warm-run under 5 minutes on RTX 5070 Ti; full-cold run ~3 h.

### Full benchmark (all subjects, V=4096, 5 seeds)

```bash
python -m eeg_bpe.run_all
```

Wall time on RTX 5070 Ti: 4–8 hours. Random seeds: `[42, 123, 456, 789, 2024]`.

### Ablation studies

Ablations A1–A15 and extension K7 (BPE+PSD ensemble) are run via:

```bash
python -m eeg_bpe.ablations --all
python -m eeg_bpe.exp6_alt_tokenization   # exp 6: alternative preprocessors
python -m eeg_bpe.exp8_csp_bpe            # CSP+BPE extension
python -m eeg_bpe.exp9_spatial_bpe        # spatial-BPE via k-means
```

Result CSVs land in `results/logs/ablation_A*.csv` and are consumed by
`scripts/generate_results_tables.py`.

### Aggregate result tables

```bash
python scripts/generate_results_tables.py
python scripts/key_findings.py
```

## Recompiling the paper

```bash
cd writing
pdflatex paper_journal.tex     # pass 1
pdflatex paper_journal.tex     # pass 2 (for cross-references)
```

Requires MiKTeX or TeX Live with `tempora`, `babel[russian,english]`, `hyperref`, `booktabs`, `enumitem`, `multicol`, `mdframed`, `fancyhdr`.

## Datasets used (all public)

| Dataset | Paradigm | Subjects | Reference |
|---|---|---|---|
| BCI-IV-2a | Motor imagery | 9 | Brunner et al. 2008 (Graz TU tech report) |
| PhysioNet-MI | Motor imagery | 109 | [DOI:10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215) |
| Sleep-EDF | Sleep staging | 20 of 78 | [DOI:10.1109/10.867928](https://doi.org/10.1109/10.867928) |
| BNCI P300 | P300 ERP | 10 | [DOI:10.1016/j.jneumeth.2007.03.005](https://doi.org/10.1016/j.jneumeth.2007.03.005) |
| Mental Arithmetic | Mental load | 36 | [DOI:10.3390/data4010014](https://doi.org/10.3390/data4010014) |
| SSVEP | Steady-state VEP | 9 | [DOI:10.1109/TBME.2017.2694818](https://doi.org/10.1109/TBME.2017.2694818) |

## Cite as

If you use this code or the findings, please cite the paper:

```bibtex
@article{nasybullin2026waveform,
  title   = {Waveform or rhythm: {BPE} tokenisation benchmark for {EEG}},
  author  = {Nasybullin, Albert A. and Kornaev, Alexey V.},
  journal = {Proceedings of the Southwest State University.
             Computer Science, Computer Engineering and Control
             = {Izvestiya Yugo-Zapadnogo gosudarstvennogo universiteta}},
  year    = {2026}
}
```

See also [`CITATION.cff`](CITATION.cff).

## Contact

- Albert Nasybullin — `a.nasibullin@innopolis.university` — [ORCID:0009-0000-7265-4378](https://orcid.org/0009-0000-7265-4378)
- Alexey Kornaev — `a.kornaev@innopolis.ru` — [ORCID:0000-0001-5121-6045](https://orcid.org/0000-0001-5121-6045)

Innopolis University, Innopolis, Republic of Tatarstan, Russia

## License

MIT — see [LICENSE](LICENSE).
