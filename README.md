# LayerProbe-DF: Audio Deepfake Detection with Frozen WavLM

**Learnable Layer Aggregation for Audio Deepfake Detection**

LayerProbe-DF is a research prototype that investigates whether combining hidden-layer representations from a frozen WavLM-Base+ model can improve audio deepfake detection.

##  Research Focus

- Layer-wise probing using Logistic Regression
- Learnable aggregation of 13 hidden-state representations
- Cross-dataset evaluation using the Fake-or-Real (FoR) dataset
- Robustness testing under noise, resampling, and compression
- Explainability through layer weights and feature visualizations

##  Methodology

- **Backbone:** `microsoft/wavlm-base-plus`
- **Backbone status:** Frozen
- **Audio sample rate:** 16 kHz
- **Clip duration:** 4 seconds
- **Feature dimensions:** 13 × 768
- **Trainable parameters:** 782
- **Baselines:** Last-layer and best-layer Logistic Regression
- **Metrics:** EER, ROC-AUC, and accuracy

##  Experimental Results

| Dataset | Method | EER ↓ | AUC ↑ | Accuracy |
|---|---|---:|---:|---:|
| ASVspoof Dev | Last layer + LR | 8.00% | 0.9725 | 91.50% |
| ASVspoof Dev | Best layer (4) + LR | 1.50% | 0.9972 | 99.00% |
| ASVspoof Dev | **LayerAgg** | **0.50%** | **0.9993** | **99.50%** |
| ASVspoof Eval (400 samples) | Last layer + LR | 4.00% | 0.9767 | 96.00% |
| ASVspoof Eval (400 samples) | Best layer (4) + LR | 2.75% | 0.9904 | 94.75% |
| ASVspoof Eval (400 samples) | **LayerAgg** | **1.25%** | **0.9988** | **97.25%** |
| FoR | Last layer + LR | 17.50% | 0.9054 | 63.50% |
| FoR | Best layer (4) + LR | 20.50% | 0.8766 | 55.00% |
| FoR | **LayerAgg** | **13.50%** | **0.9480** | **71.25%** |

*Results are from selected balanced subsets, not full-dataset evaluations.*

##  Datasets

- **ASVspoof 2019 LA:** Training, development, and a 400-sample official evaluation subset.
- **Fake-or-Real (FoR):** A balanced 400-sample subset for cross-dataset testing.

**Label convention:** `0 = Real (bonafide)`, `1 = Fake (spoof)`.

##  Analysis and Visualizations

The project includes:
- Layer-wise performance comparisons
- ROC curves
- Learned layer weights
- Training-loss plots
- t-SNE visualizations
- Audio-corruption robustness results
- Temporal window-level predictions

##  Limitations

- Experiments use small balanced subsets.
- The official evaluation results cover only 400 samples.
- Results are based on a single run.
- Noise robustness results require further validation.
- Layer weights do not establish causal explanations.
- No state-of-the-art performance claim is made.

##  Future Work

- Evaluate on larger dataset splits.
- Repeat experiments with multiple random seeds.
- Improve noise robustness evaluation.
- Test additional speech foundation models.
- Investigate cross-dataset calibration and generalization.

##  Repository Contents

- `LayerProbe_DF_Experiments.ipynb` — Google Colab notebook
- `LayerProbe_DF_Research_Report.pdf` — Research report
- `results_main.csv` — Main experimental results
- `results_robustness.csv` — Robustness results
- `results_official_eval.csv` — Official evaluation subset results
- `figs/` — Experimental visualizations