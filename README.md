# DLP Benchmark: Indonesian PII Detection Across Paradigms

A comprehensive benchmarking framework for evaluating Data Loss Prevention (DLP) systems on Indonesian Personally Identifiable Information (PII) detection. This benchmark compares three paradigms—Pattern-Matching (Regex), Discriminative (Fine-tuned BERT), and Reasoning (LLM)—across linguistically diverse Indonesian text styles.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Workflow](#workflow)
6. [Dataset Generation](#dataset-generation)
7. [Model Training](#model-training)
8. [Benchmarking](#benchmarking)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Output Files](#output-files)
11. [Advanced Features](#advanced-features)
12. [Methodological Rigor](#methodological-rigor)
13. [Troubleshooting](#troubleshooting)
14. [Citation](#citation)

---

## Overview

### Research Question

**How do different AI paradigms perform on Indonesian PII detection across varying linguistic formality levels?**

### Key Features

- **Three Paradigms Evaluated:**
  - Pattern-Matching (Regex with keyword disambiguation)
  - Discriminative (Fine-tuned IndoBERT-NER)
  - Reasoning (Llama-3-8B with 4-bit quantization)

- **Six PII Types Detected:**
  - NIK (National ID, 16 digits)
  - Phone Numbers (Indonesian format)
  - Credit Card Numbers (13-19 digits)
  - Bank Account Numbers (8-15 digits)
  - Email Addresses
  - Personal Names

- **Three Linguistic Styles:**
  - Formal (standard Indonesian)
  - Code-Mixed (Indonesian + English)
  - Informal (colloquial, phonological reductions)

- **Rigorous Evaluation:**
  - Precision, Recall, F1 scores
  - Bootstrap confidence intervals
  - Paired permutation tests
  - Effect size analysis (Cohen's d, Cliff's Delta)
  - Comprehensive latency benchmarking

---

## Architecture

### Project Structure

```
dlp-v2/
├── dataset.py                      # Dataset generation with linguistic transformations
├── train_bert_ner.py              # BERT fine-tuning script
├── prompts_config.py              # Externalized LLM prompts
├── evaluation_utils.py            # Advanced evaluation utilities
├── dlp_benchmark_v2.py            # Main benchmarking pipeline
│
├── BERT_TRAINING_README.md        # BERT training documentation
├── PROMPT_ABLATION_README.md      # Prompt ablation guide
├── ADVANCED_EVALUATION_README.md  # Advanced evaluation guide
├── README.md                      # This file
│
├── bert_dlp_finetuned/            # Fine-tuned BERT model (generated)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
│
└── results/                       # Output directory (generated)
    ├── dataset_formal.csv
    ├── dataset_code_mixed.csv
    ├── dataset_informal.csv
    ├── regex_results.csv
    ├── bert_results.csv
    ├── llm_results.csv
    ├── full_report.csv
    ├── significance_report.csv
    ├── latency_comprehensive.csv
    ├── effect_sizes.csv
    └── ...
```

### Module Dependencies

```
dlp_benchmark_v2.py
    ├── dataset.py              (dataset generation)
    ├── prompts_config.py       (LLM prompts)
    ├── evaluation_utils.py     (advanced metrics)
    └── train_bert_ner.py       (BERT training, run separately)
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- 32GB+ RAM

### Setup

```bash
# Clone repository
git clone <repository-url>
cd dlp-v2

# Install dependencies
pip install faker transformers bitsandbytes accelerate torch pandas scikit-learn tqdm scipy

# Upgrade Hugging Face Hub
pip install --upgrade huggingface_hub

# Set Hugging Face token (for Llama-3 access)
export HF_TOKEN="your_huggingface_token_here"
```

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16GB) | NVIDIA A100 (40GB) |
| RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |
| CPU | 4 cores | 8+ cores |

---

## Quick Start

### End-to-End Execution

```python
# Run the complete benchmark pipeline
python dlp_benchmark_v2.py
```

This will:
1. Generate datasets (formal, code-mixed, informal)
2. Fine-tune BERT on the training data
3. Run all three paradigms on evaluation data
4. Compute comprehensive metrics
5. Generate reports and visualizations

**Estimated Runtime:** 4-6 hours on A100 GPU

---

## Workflow

### Phase 1: Dataset Generation

**Module:** `dataset.py`

**Process:**
1. Generate 5,000 synthetic identities using Faker
2. Create 15,000 prompts (5,000 per style)
3. Apply linguistically grounded transformations:
   - **Formal:** Standard Indonesian grammar
   - **Code-Mixed:** Indonesian + English mixing
   - **Informal:** Phonological reductions, colloquialisms

**Output:**
- `ground_truth.csv`
- `prompt_dataset.csv`
- `eval_sample.csv`
- `linguistic_validation.csv`
- `validation_sample.csv`

**Key Features:**
- 120 diverse templates (40 per style)
- Real Indonesian phonological patterns (e.g., "tolong" → "tlg")
- Contextual code-switching
- Informal pronouns and particles

**Example Transformations:**

| Style | Example |
|-------|---------|
| Formal | "Nomor telepon saya adalah 081234567890" |
| Code-Mixed | "My phone number adalah 081234567890" |
| Informal | "No hp gw 081234567890" |

---

### Phase 2: Model Training

**Module:** `train_bert_ner.py`

**Process:**
1. Load IndoBERT base model
2. Define 13-label schema:
   - O (Outside)
   - B-NIK, I-NIK
   - B-PHONE, I-PHONE
   - B-CREDIT_CARD, I-CREDIT_CARD
   - B-EMAIL, I-EMAIL
   - B-NAME, I-NAME
   - B-BANK, I-BANK
3. Tokenize and align labels with subword tokens
4. Fine-tune for 3 epochs
5. Save to `./bert_dlp_finetuned/`

**Training Configuration:**
```python
training_args = TrainingArguments(
    output_dir="./bert_dlp_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

**Output:**
- Fine-tuned model in `./bert_dlp_finetuned/`
- Training logs and metrics

**Runtime:** ~1-2 hours on A100 GPU

---

### Phase 3: Benchmarking

**Module:** `dlp_benchmark_v2.py`

#### 3.1 Paradigm A: Pattern-Matching (Regex)

**Implementation:**
```python
def regex_detect(text: str) -> dict:
    # NIK: 16-digit pattern with keyword disambiguation
    # Phone: Indonesian format (08xx, +62)
    # Credit Card: 13-19 digits with keyword context
    # Bank: 8-15 digits with keyword context
    # Email: Standard email regex
    # Name: Not supported (open-vocabulary)
```

**Features:**
- Keyword-based disambiguation (NIK vs CC)
- Context-aware detection
- CPU-based (no GPU required)

**Limitations:**
- Cannot detect names (open-vocabulary entity)
- Sensitive to formatting variations
- No semantic understanding

---

#### 3.2 Paradigm B: Discriminative (IndoBERT-NER)

**Implementation:**
```python
def bert_detect(text: str, ner_pipe) -> dict:
    # Pure NER detection (NO HEURISTICS)
    # Uses fine-tuned model with 13-label schema
    # Aggregates subword tokens into entities
```

**Features:**
- Learned patterns from training data
- Handles formatting variations
- Detects all 6 PII types including names
- GPU-accelerated

**Advantages:**
- Robust to noise and typos
- Contextual understanding
- Fast inference (batch processing)

---

#### 3.3 Paradigm C: Reasoning (Llama-3-8B)

**Implementation:**
```python
def llm_detect(text: str, tokenizer, model, prompt_id: str) -> dict:
    # LLM-based detection with structured output
    # Uses externalized prompts from prompts_config.py
    # 4-bit quantization for memory efficiency
```

**Features:**
- Zero-shot reasoning
- Semantic understanding
- Handles ambiguous cases
- Configurable prompts

**Prompt Variants:**
- `detailed_v1`: Comprehensive instructions
- `minimal_v1`: Concise instructions
- `few_shot_v1`: With examples

**Advantages:**
- No training required
- Generalizes to unseen patterns
- Explainable (can provide reasoning)

---

### Phase 4: Evaluation

**Module:** `evaluation_utils.py` + `dlp_benchmark_v2.py`

#### 4.1 Detection Metrics

**Per-Sample Computation:**
```python
# For each sample:
gt_set = {normalized_ground_truth_values}
det_set = {normalized_detected_values}

tp = len(gt_set & det_set)      # True Positives
fp = len(det_set - gt_set)      # False Positives
fn = len(gt_set - det_set)      # False Negatives

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
```

**Normalization:**
- Numeric PII: Strip spaces, dashes, leading zeros
- Email: Lowercase, strip whitespace
- Name: Normalize whitespace, lowercase

**Aggregation:**
- Per-PII-type metrics
- Per-style metrics
- Overall macro-average

---

#### 4.2 Statistical Rigor

**Bootstrap Confidence Intervals:**
```python
# 1,000 bootstrap resamples
# 95% confidence intervals for recall
# Per-PII-type, per-style
```

**Paired Permutation Tests:**
```python
# H0: Recall(A) = Recall(B)
# 10,000 permutations
# Two-tailed p-value
# Significance threshold: p < 0.05
```

**Effect Size Analysis:**
```python
# Cohen's d (paired):
differences = [score_a[i] - score_b[i] for i in samples]
cohens_d = mean(differences) / std(differences)

# Cliff's Delta (non-parametric):
dominance = count(a > b for all pairs)
cliffs_delta = (dominance - non_dominance) / total_pairs

# Bootstrap CI for effect sizes
# Bonferroni correction for multiple comparisons
```

---

#### 4.3 Latency Benchmarking

**Deployment Scenarios:**

| Scenario | Batch Size | Description |
|----------|-----------|-------------|
| Interactive | 1 | Real-time single requests |
| Cold Start | 1 | First request after idle |
| Streaming | 1 | Sustained single requests |
| Batch Small | 8 | Small batch processing |
| Batch Medium | 32 | High-throughput processing |

**Measurement Protocol:**
1. **Adaptive Warm-up:**
   - Run until latency converges (CV < 10%)
   - Typically 5-20 iterations for GPU models
   
2. **Proper GPU Synchronization:**
   ```python
   torch.cuda.synchronize()  # Before timing
   output = model(batch)
   torch.cuda.synchronize()  # After inference
   ```

3. **Metrics Collected:**
   - Mean, median, p50, p95, p99 latency
   - Throughput (samples/sec)
   - Warm-up iterations needed

---

## Evaluation Metrics

### Detection Performance

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Accuracy of detections |
| **Recall** | TP / (TP + FN) | Coverage of ground truth |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean |

### Statistical Significance

| Test | Purpose | Output |
|------|---------|--------|
| **Bootstrap CI** | Uncertainty quantification | 95% CI for recall |
| **Permutation Test** | Pairwise comparison | p-value, significance flag |
| **Effect Size** | Magnitude of difference | Cohen's d, interpretation |

### Latency

| Metric | Unit | Description |
|--------|------|-------------|
| **Mean Latency** | ms | Average per-sample time |
| **p95 Latency** | ms | 95th percentile (tail latency) |
| **Throughput** | samples/sec | Batch processing speed |

---

## Output Files

### Dataset Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `dataset_formal.csv` | 5,000 | 15 | Formal Indonesian prompts |
| `dataset_code_mixed.csv` | 5,000 | 15 | Code-mixed prompts |
| `dataset_informal.csv` | 5,000 | 15 | Informal prompts |

**Columns:**
- `identity_id`: Unique identifier
- `style`: Linguistic style
- `prompt`: Generated text
- `ground_truth_nik`, `ground_truth_phone`, etc.: Ground truth values
- `has_nik`, `has_phone`, etc.: Boolean flags

---

### Benchmark Results

| File | Description |
|------|-------------|
| `regex_results.csv` | Regex detection results |
| `bert_results.csv` | BERT detection results |
| `llm_results.csv` | LLM detection results |

**Columns:**
- Detection results: `detected_niks`, `detected_phones`, etc.
- Ground truth: `ground_truth_nik`, `ground_truth_phone`, etc.
- Metadata: `identity_id`, `style`, `latency_ms`

---

### Evaluation Reports

| File | Description |
|------|-------------|
| `full_report.csv` | Precision, recall, F1 for all paradigms × PII types × styles |
| `significance_report.csv` | Pairwise permutation test results |
| `robustness_report.csv` | Performance degradation across styles |
| `latency_report.csv` | Basic latency summary |
| `latency_comprehensive.csv` | Multi-scenario latency analysis |
| `effect_sizes.csv` | Effect size analysis with statistical tests |
| `effect_sizes_table.tex` | LaTeX table for publication |
| `prompt_log.csv` | Full prompt text for reproducibility |

---

## Advanced Features

### 1. Prompt Ablation

**Purpose:** Test LLM robustness across prompt variants

**Usage:**
```python
ablation_results = run_llm_prompt_ablation(df_eval, tokenizer, model)
stability_report = compare_prompt_stability(ablation_results)
```

**Output:**
- Recall variance across prompts
- Coefficient of variation per PII type
- Prompt sensitivity analysis

**Documentation:** See `PROMPT_ABLATION_README.md`

---

### 2. Comprehensive Latency Benchmarking

**Purpose:** Evaluate performance across deployment scenarios

**Usage:**
```python
paradigms = {
    "Pattern-Matching (Regex)": regex_detect,
    "Discriminative (IndoBERT-NER)": lambda x: bert_detect(x, ner_pipe),
    "Reasoning (Llama-3-8B)": lambda x: llm_detect(x, tokenizer, model)
}

df_latency = run_comprehensive_latency_benchmark(
    df_eval=df_eval,
    paradigms=paradigms,
    output_dir="./results"
)
```

**Features:**
- Adaptive warm-up
- Multiple deployment scenarios
- GPU synchronization
- Throughput measurement

**Documentation:** See `ADVANCED_EVALUATION_README.md`

---

### 3. Effect Size Analysis

**Purpose:** Quantify magnitude of performance differences

**Usage:**
```python
df_effect_sizes = generate_effect_size_report(
    df_regex=df_regex,
    df_bert=df_bert,
    df_llm=df_llm,
    output_dir="./results"
)

latex_table = generate_effect_size_latex_table(
    df_effect_sizes=df_effect_sizes,
    output_dir="./results"
)
```

**Features:**
- Cohen's d with bootstrap CI
- Cliff's Delta (non-parametric)
- Bonferroni correction
- Sample size diagnostics

**Documentation:** See `ADVANCED_EVALUATION_README.md`

---

## Methodological Rigor

### Addressing Reviewer Concerns

This benchmark was designed to address specific methodological concerns raised during peer review:

#### MW1: Dataset Validity
**Concern:** Synthetic noise ≠ real Indonesian informal language

**Solution:**
- Linguistically grounded transformations
- Real phonological patterns (e.g., "gimana" → "gmn")
- Contextual code-switching
- 120 diverse templates

**File:** `dataset.py`

---

#### MW2: Real Model Training
**Concern:** Pretrained NER + heuristics ≠ fair comparison

**Solution:**
- Fine-tuned BERT with explicit 13-label schema
- End-to-end trainable pipeline
- No heuristic leakage

**File:** `train_bert_ner.py`

---

#### Red Flag 1: Heuristic Leakage
**Concern:** Mixing learned predictions with hand-crafted rules

**Solution:**
- Pure BERT NER detection (no post-processing)
- Separate rule-based baseline (Regex)
- Explicit documentation of each paradigm's capabilities

**File:** `dlp_benchmark_v2.py` (lines 185-225)

---

#### Red Flag 2: LLM Reproducibility
**Concern:** Hardcoded prompts, no logging

**Solution:**
- Externalized prompts in `prompts_config.py`
- Versioned prompt variants
- Full prompt logging to CSV
- Prompt ablation framework

**Files:** `prompts_config.py`, `PROMPT_ABLATION_README.md`

---

#### Red Flag 3: Incomplete Evaluation
**Concern:** Only 3 of 6 PII types evaluated

**Solution:**
- Evaluate ALL 6 PII types
- Explicit documentation: Regex does not support NAME (open-vocabulary)
- Separate metrics per PII type

**File:** `dlp_benchmark_v2.py` (evaluation functions)

---

#### mW1: Missing Precision/F1
**Concern:** Only recall reported

**Solution:**
- Set-based matching for correct TP/FP/FN counting
- Precision, recall, F1 for all paradigms
- Per-PII-type, per-style metrics

**File:** `dlp_benchmark_v2.py` (lines 652-737)

---

#### MW3: Invalid Latency Measurement
**Concern:** No batching, no warm-up, not comparable

**Solution:**
- Adaptive warm-up with convergence detection
- Multiple deployment scenarios
- Proper GPU synchronization
- Throughput measurement with batching

**File:** `evaluation_utils.py`, `ADVANCED_EVALUATION_README.md`

---

#### mW3: Missing Effect Size
**Concern:** Statistical significance ≠ practical significance

**Solution:**
- Cohen's d with bootstrap CI
- Cliff's Delta (non-parametric)
- Bonferroni correction
- Sample size diagnostics

**File:** `evaluation_utils.py`, `ADVANCED_EVALUATION_README.md`

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in `train_bert_ner.py`:
  ```python
  per_device_train_batch_size=8  # Instead of 16
  ```
- Use gradient accumulation:
  ```python
  gradient_accumulation_steps=2
  ```
- Enable gradient checkpointing:
  ```python
  model.gradient_checkpointing_enable()
  ```

---

#### 2. Llama-3 Access Denied

**Symptom:**
```
HTTPError: 401 Unauthorized
```

**Solutions:**
- Request access to Llama-3 on Hugging Face
- Set HF_TOKEN environment variable:
  ```bash
  export HF_TOKEN="your_token_here"
  ```
- Or set in code:
  ```python
  os.environ['HF_TOKEN'] = "your_token_here"
  ```

---

#### 3. Slow Dataset Generation

**Symptom:**
Dataset generation takes > 30 minutes

**Solutions:**
- Use pre-generated datasets (if available)
- Reduce `NUM_IDENTITIES`:
  ```python
  NUM_IDENTITIES = 1_000  # Instead of 5_000
  ```
- Disable validation:
  ```python
  validate_dataset=False
  ```

---

#### 4. BERT Training Fails

**Symptom:**
```
ValueError: Label schema mismatch
```

**Solutions:**
- Ensure dataset has all 6 PII types
- Check label alignment in tokenization
- Verify `id2label` and `label2id` mappings

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
coming soon
```

---

## License

Coming soon

---

## Contact

For questions, issues, or contributions:
- coming soon

---

## Acknowledgments

- **IndoBERT:** Pre-trained Indonesian BERT model
- **Llama-3:** Meta's large language model
- **Faker:** Synthetic data generation library
- **Hugging Face:** Transformers library and model hub

---

## Appendix: File Descriptions

### Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| `dataset.py` | 880 | Linguistically grounded dataset generation |
| `train_bert_ner.py` | 456 | BERT fine-tuning with 13-label schema |
| `prompts_config.py` | 200 | Externalized LLM prompts |
| `evaluation_utils.py` | 738 | Advanced evaluation utilities |
| `dlp_benchmark_v2.py` | 1,482 | Main benchmarking pipeline |

### Documentation

| File | Lines | Purpose |
|------|-------|---------|
| `BERT_TRAINING_README.md` | 189 | BERT training guide |
| `PROMPT_ABLATION_README.md` | 200 | Prompt ablation guide |
| `ADVANCED_EVALUATION_README.md` | 520 | Advanced evaluation guide |
| `README.md` | This file | Complete documentation |

**Total:** ~4,665 lines of production-ready code + documentation

---

## Version History

- **v2.0** (Current): Complete rewrite with methodological rigor
  - Linguistically grounded dataset
  - Fine-tuned BERT (no heuristics)
  - Externalized prompts
  - Comprehensive evaluation (precision, F1, effect size, latency)
  
- **v1.0** (Initial): Basic benchmark with reviewer concerns

---

**Last Updated:** 2024-04-29
