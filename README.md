### Dataset Generation

**Module:** `dataset_generation_dlp.ipynb`

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
