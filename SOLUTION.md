# Hallucination Detection — Solution Report

## Reproducibility

### Environment

Developed in Google Colab with a T4 GPU. All dependencies are in `requirements.txt`.

### Commands to reproduce

```bash
git clone https://github.com/SmirnovEA/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
The script loads Qwen2.5-0.5B, extracts hidden states from 689 samples, applies 5-fold stratified cross-validation with a 20% held-out test set, and saves results.json and predictions.csv. Test AUROC averaged over 5 folds should reproduce to 74.5–74.9%.

Expected outputs
results.json — per-fold metrics and summary

predictions.csv — competition test set predictions

Final Solution Description
Overview
Only the data splitting strategy was modified. Feature extraction and classifier architecture remain as in the provided baseline. Systematic experiments showed that every modification to the model or features degraded performance on this dataset of 689 samples.

Components
Aggregation (aggregation.py). The last real token of the final transformer layer from Qwen2.5-0.5B produces an 896-dimensional feature vector. Multi-layer concatenation and mean pooling were tested and rejected.

Probe classifier (probe.py). A two-layer MLP with hidden size 256, ReLU activation, and a single logit output. Binary cross-entropy loss with positive class weighting handles the 70/30 imbalance. Adam optimizer, learning rate 1e-3, 200 epochs. The decision threshold is tuned on the validation split to maximize F1.

Data splitting (splitting.py). Stratified 5-fold cross-validation with a 20% held-out test set replaces the original single random split. From 689 samples, 138 are held out. The remaining 551 are divided into 5 stratified folds, each alternating as validation while the other 4 form the training set. Metrics are averaged across folds.

What Contributed Most
Replacing the single split with 5-fold cross-validation was the only beneficial change. Single-split AUROC varied by 4–6 points depending on the random seed. Averaging removes this variance. All model modifications introduced overfitting without improving the metric.

Experiments and Failed Attempts
Each candidate was evaluated under identical 5-fold cross-validation and compared to the 74.5–74.9% baseline.

Multi-layer aggregation. Concatenating hidden states from layers 8, 16, and 23 expanded the feature dimension from 896 to 2688. With 481 training examples the model overfit severely and Test AUROC dropped to 57–62%.

Token pooling alternatives. Weighted mean pooling over all real tokens caused a collapse to the majority-class baseline (Test AUROC ~57%).

Geometric features. Eleven hand-crafted features including L2 norms, token variance, cosine similarities, and sequence length were added. In all cases they left the metric unchanged or reduced it.

Dropout and smaller architectures. Adding dropout (0.2–0.5) to networks with hidden sizes 64–128 lowered Test AUROC to 69–72%.

Hidden layer size. Networks with hidden sizes of 64, 128, 256, and 512 were compared. Results: 73.9%, 72.1% (with dropout), 74.5–74.9%, and 74.4%. The default size of 256 gave the best result.

Logistic regression. A linear probe achieved 69–70% AUROC, confirming the need for a non-linear hidden layer.

Test set size. Increasing the held-out set from 15% to 20% reduced fold-to-fold variance and was adopted.

The experiments show that the default pipeline extracts near-optimal representations. The only improvement is in the evaluation methodology.
