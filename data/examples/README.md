# Example Data Files

This directory contains **synthetic example data** to demonstrate the expected format for the clinical feature extraction pipeline. These are NOT real patient data.

The framework is **domain-agnostic** and works with any type of clinical text (discharge summaries, pathology reports, progress notes, etc.).

## Purpose

These example files show the structure and format expected by the pipeline scripts, allowing you to:
- Understand the required columns and data types
- Test the pipeline code without access to real clinical data
- Develop and debug new features safely

## Files

### `discharge_notes.csv`
Example clinical notes (discharge summaries).

**Columns:**
- `hadm_id` (int): Hospital admission ID (primary key)
- `discharge_text` (str): Full discharge summary text

**Use case:** Demonstrates the expected format for any clinical text (discharge summaries, pathology reports, progress notes, etc.)

**Note:** Your clinical notes file can have different column names and ID types. Just update `config/pipeline_config.py` accordingly:
- Set `NOTES_COL` to your text column name
- Set `INDEX_COL` to your ID column name

---

### `structured_features.csv`
Example structured clinical features and outcomes.

**Columns:**
- `hadm_id` (int): Hospital admission ID (matches discharge_notes.csv)
- Demographics: `age`, `gender`
- Vitals: `heart_rate`, `sbp`, `spo2`, `temperature`, `bmi`
- Labs: `bicarbonate`, `creatinine`, `hemoglobin`, `inr`, `platelet`, `potassium`, `wbc`, `sodium`, `nt_probnp`, `troponin_t`
- Comorbidities: Binary indicators (0/1) for various conditions
- Outcome: `death_30_days` (0/1)

**Use case:** Baseline structured features for model evaluation (optional)

---

## Important Notes

1. **Synthetic Data**: All names, dates, and clinical details are completely fabricated
2. **Format Only**: These examples show structure, not realistic clinical distributions
3. **Small Sample**: Only 3 examples per file for demonstration purposes
4. **Not for Training**: Do not use for actual model training or validation
5. **Flexible Format**: Your data can have different column names - just update the config file

## Using Example Data

To test the pipeline with example data, modify the config file paths:

```python
# In config/pipeline_config.py
NOTES_FILE_PATH = os.path.join(_DATA_DIR, 'examples', 'discharge_notes.csv')
```

## Real Data

Real clinical data must be:
- Stored outside this repository
- Protected according to HIPAA/institutional guidelines
- De-identified per HIPAA Safe Harbor or Expert Determination
- Never committed to version control

See [DATA_SETUP.md](../../DATA_SETUP.md) for instructions on setting up real data files.
