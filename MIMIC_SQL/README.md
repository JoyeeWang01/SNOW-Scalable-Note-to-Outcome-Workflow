# MIMIC-IV Cohort Selection SQL

This directory contains SQL queries to reproduce the HFpEF (Heart Failure with preserved Ejection Fraction) cohort from MIMIC-IV used in the paper.

## Quick Checklist

Before you can run these queries, ensure you have:

- [ ] PhysioNet account created
- [ ] CITI training completed and certificate uploaded
- [ ] Access approved for **MIMIC-IV v3.1** (https://physionet.org/content/mimiciv/3.1/)
- [ ] Access approved for **MIMIC-IV-Note v2.2** (https://physionet.org/content/mimic-iv-note/2.2/)
- [ ] PhysioNet credentials linked to Google Cloud
- [ ] Google Cloud project created with billing enabled
- [ ] Verified BigQuery access to both datasets

---

## Prerequisites

Before running these SQL queries, you **must** complete the following steps:

### 1. Complete PhysioNet Credentialing

You need access to **two separate PhysioNet datasets**:

**Required Dataset 1: MIMIC-IV v3.1** (Core clinical data)
- **URL**: https://physionet.org/content/mimiciv/3.1/
- **Contains**: Hospital admissions, diagnoses, labs, vitals, medications

**Required Dataset 2: MIMIC-IV-Note v2.2** (Clinical notes)
- **URL**: https://physionet.org/content/mimic-iv-note/2.2/
- **Contains**: Discharge summaries, radiology reports, ECG reports

**Steps to get access:**

1. **Create PhysioNet account**: https://physionet.org/register/
2. **Complete CITI training**:
   - Go to: https://physionet.org/about/citi-course/
   - Complete "Data or Specimens Only Research" course
   - Upload certificate to PhysioNet profile
3. **Sign Data Use Agreement (DUA)** for **both datasets**:
   - Sign DUA for MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
   - Sign DUA for MIMIC-IV-Note v2.2: https://physionet.org/content/mimic-iv-note/2.2/
4. **Wait for approval** (typically 1-3 business days per dataset)

### 2. Set Up Google BigQuery Access

After getting PhysioNet access:

1. **Link PhysioNet to Google Cloud**:
   - Follow instructions: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/
   - You'll need to link your PhysioNet credentials to your Google account

2. **Create Google Cloud project** with billing enabled:
   ```bash
   # Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
   gcloud projects create your-project-name
   gcloud config set project your-project-name
   # Enable billing in Cloud Console
   ```

3. **Verify BigQuery access**:
   ```sql
   -- Test query in BigQuery Console (should return ~200k rows)
   SELECT COUNT(*) FROM `physionet-data.mimiciv_3_1_hosp.admissions`;

   -- Test notes access (should return ~300k rows)
   SELECT COUNT(*) FROM `physionet-data.mimiciv_note.discharge`;
   ```

### 3. Required BigQuery Datasets

Once approved, you'll have access to these BigQuery datasets:

**From MIMIC-IV v3.1:**
- `physionet-data.mimiciv_3_1_hosp` - Hospital data (diagnoses, procedures, labs)
- `physionet-data.mimiciv_3_1_icu` - ICU stays and measurements
- `physionet-data.mimiciv_3_1_derived` - Derived tables (Charlson, vitals, etc.)
- `physionet-data.mimiciv_3_1_ed` - Emergency Department visits

**From MIMIC-IV-Note v2.2:**
- `physionet-data.mimiciv_note.discharge` - Discharge summaries (used in Step 2)
- `physionet-data.mimiciv_note.radiology` - Radiology reports (not used here)
- `physionet-data.mimiciv_note.ecg` - ECG reports (not used here)

**⚠️ IMPORTANT**: Both datasets are required. The SQL queries will fail if you only have access to one.

## Cohort Selection Overview

The cohort selection follows a **3-step process**:

1. **Cohort Selection** (`01_cohort_selection.sql`) - Select HFpEF patients
2. **Add Discharge Notes** (`02_discharge_notes.sql`) - Join clinical notes
3. **Gather Structured Features** (`03_structured_features.sql`) - Gather baseline features and outcomes

## Execution Instructions

### Step 1: Cohort Selection

**File:** `01_cohort_selection.sql`

**What it does:**
- Identifies adult patients (≥18 years) with HFpEF as primary diagnosis (ICD-9: 428.30-428.33, ICD-10: I50.30-I50.33)
- Selects **index admission** (first HFpEF admission per patient)
- Excludes patients who died in-hospital (hospital_expire_flag = 0)

**Output table:** Your cohort table (update the table name in the SQL)

**To run:**
1. Open Google BigQuery Console
2. Create a new dataset for your project (e.g., `your_project.mimic_cohort`)
3. Update the output table name on line 2:
   ```sql
   CREATE OR REPLACE TABLE `your_project.your_dataset.hfpef_cohort` AS
   ```
4. Run the query

**Expected result:** ~2,400 unique patients (varies slightly by MIMIC-IV version)

---

### Step 2: Add Discharge Notes

**File:** `02_discharge_notes.sql`

**What it does:**
- Joins discharge summaries from `mimiciv_note.discharge` to the cohort
- Concatenates multiple discharge notes per admission (if any) with separators
- Orders notes by chart time (most recent first)

**Prerequisites:** Step 1 completed

**Output table:** Cohort with discharge notes

**To run:**
1. Update line 1 with your output table name:
   ```sql
   CREATE OR REPLACE TABLE `your_project.your_dataset.hfpef_with_notes` AS
   ```
2. Update line 12 with your cohort table from Step 1:
   ```sql
   FROM `your_project.your_dataset.hfpef_cohort` AS h
   ```
3. Run the query

**Expected result:** Same number of patients as Step 1, now with `discharge_text` column

---

### Step 3: Gather Structured Features

**File:** `03_structured_features.sql`

**What it does:**
Gathers baseline structured features for each patient admission:

**Demographics:**
- Age at admission
- Gender (binary: 0=Female, 1=Male)

**Outcomes:**
- `death_30_days` - 30-day mortality after discharge
- `death_1_year` - 1-year mortality after discharge

**Vital Signs:**
- Heart rate, systolic BP, oxygen saturation, temperature (mean/min/max)
- BMI (from OMR or ICU measurements)
- Priority: ED vitals → ED triage → ICU first-day vitals

**Laboratory Values** (averaged over entire admission):
- `bicarbonate`, `creatinine`, `hemoglobin`, `inr`
- `platelet_count`, `potassium`, `wbc_count`, `sodium`
- `ntprobnp` (NT-proBNP), `troponin` (Troponin T)

**Comorbidities** (Charlson comorbidity index components):
- 15 binary indicators (0/1) for conditions like:
  - Acute MI, peripheral vascular disease, cerebrovascular disease
  - Dementia, COPD, rheumatoid disease, peptic ulcer disease
  - Diabetes (with/without complications), liver disease
  - Renal disease, cancer, etc.

**Cardiovascular Diagnoses:**
- `HT` - Hypertension
- `CAD` - Coronary artery disease
- `PH` - Pulmonary hypertension
- `AF` - Atrial fibrillation

**Prerequisites:** Step 2 completed

**Output:** Final feature table (can export to CSV for analysis)

**To run:**
1. Update line 8 with your table from Step 2:
   ```sql
   FROM `your_project.your_dataset.hfpef_with_notes` c
   ```
2. This query uses a `WITH` clause and doesn't create a table by default
3. **Option A:** Export results directly to CSV:
   - Run the query
   - Click "Save Results" → "CSV (local file)"
   - Save as `data/structured_features.csv`

4. **Option B:** Save as a BigQuery table (add at the beginning):
   ```sql
   CREATE OR REPLACE TABLE `your_project.your_dataset.hfpef_features` AS
   WITH coh AS (
     ...
   ```

**Expected result:** Same number of rows as Step 2, with ~50 feature columns + `discharge_text`

---

## Data Export

After running all three steps, export your final data:

### Export Discharge Notes

```bash
# Run 02_discharge_notes.sql, then export:
bq extract --destination_format=CSV \
  your_project:your_dataset.hfpef_with_notes \
  gs://your-bucket/discharge_notes.csv

# Download from GCS
gsutil cp gs://your-bucket/discharge_notes.csv data/discharge_notes.csv
```

### Export Structured Features

```bash
# Run 03_features.sql, then export (if saved as table):
bq extract --destination_format=CSV \
  your_project:your_dataset.hfpef_features \
  gs://your-bucket/structured_features.csv

# Download from GCS
gsutil cp gs://your-bucket/structured_features.csv data/structured_features.csv
```

**Note:** Ensure exported files contain:
- `discharge_notes.csv`: Columns `hadm_id`, `discharge_text`
- `structured_features.csv`: All features from Step 3

---

## Important Notes

### MIMIC-IV Version

This SQL is designed for **MIMIC-IV v3.1**. If using a different version:
- Update table references (e.g., `mimiciv_3_1_hosp` → `mimiciv_4_0_hosp`)
- Check for schema changes in MIMIC documentation

### Custom Table Names

The SQL files contain placeholder table names (e.g., `astute-curve-441706-n6.2400_patients.*`). **You MUST update these** to match your Google Cloud project and dataset:

```sql
-- Replace this pattern throughout:
`astute-curve-441706-n6.2400_patients.TABLE_NAME`

-- With your project and dataset:
`your-gcp-project.your_dataset_name.TABLE_NAME`
```

### BigQuery Costs

Running these queries on MIMIC-IV processes several GB of data:
- Step 1: ~5 GB processed (~$0.025)
- Step 2: ~10 GB processed (~$0.050)
- Step 3: ~50 GB processed (~$0.250)

**Total estimated cost:** ~$0.33 USD (as of 2024, verify current BigQuery pricing)

### PHI/PII Considerations

MIMIC-IV is **de-identified** but still contains clinical notes. Ensure you:
- Have completed required CITI training
- Follow your institution's IRB requirements
- Do NOT re-identify patients
- Store data securely (not in public repositories)

### Troubleshooting

**Problem:** `Access Denied` or `Not found: Dataset physionet-data:mimiciv_3_1_hosp`

**Solution:**
- Verify you have been **approved** for both PhysioNet datasets (check your email)
- Ensure you've **linked your PhysioNet credentials to Google Cloud**: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/
- Wait 10-15 minutes after linking for permissions to propagate
- Try running the test queries in the Prerequisites section to verify access

**Problem:** `Access Denied` for `physionet-data.mimiciv_note.discharge`

**Solution:**
- This means you have MIMIC-IV v3.1 access but **NOT** MIMIC-IV-Note v2.2 access
- You must separately request access to: https://physionet.org/content/mimic-iv-note/2.2/
- Sign the DUA for MIMIC-IV-Note v2.2 and wait for approval (1-3 business days)

**Problem:** `Table not found` errors after getting access

**Solution:**
- Verify you're using the correct dataset version references (e.g., `mimiciv_3_1_hosp` not `mimiciv_hosp`)
- Update your table names in the SQL files (marked with TODO comments)

**Problem:** Different patient counts than expected (~2,400)

**Solution:**
- MIMIC-IV is updated periodically; expect slight variations (±100 patients)
- Verify you're using MIMIC-IV v3.1 (not v2.2 or v4.0)
- Check for NULL values in key fields if counts are drastically different

**Problem:** Missing columns in derived tables (e.g., `charlson`, `first_day_vitalsign`)

**Solution:**
- Some derived tables may not be available in all MIMIC versions
- Verify table schemas:
  ```sql
  SELECT column_name, data_type
  FROM `physionet-data.mimiciv_3_1_derived.INFORMATION_SCHEMA.COLUMNS`
  WHERE table_name = 'charlson';
  ```

**Problem:** Query times out or is very slow

**Solution:**
- BigQuery can handle these queries; timeouts are rare
- Ensure you're not running queries in a free tier project (limited resources)
- Step 3 processes ~50 GB; ensure adequate BigQuery quota

---

## Expected Output Summary

| Step | Output | Rows | Columns | Key Fields |
|------|--------|------|---------|-----------|
| 1. Cohort Selection | HFpEF cohort | ~2,400 | 12 | `subject_id`, `hadm_id`, demographics |
| 2. Add Notes | Cohort + notes | ~2,400 | 13 | + `discharge_text` |
| 3. Gather Structured Features | Final dataset | ~2,400 | ~50 | All features + outcome |

---

## Citation

If you use this cohort selection methodology, please cite:

```bibtex
@article{wang2025agentbased,
  title={Agent-Based Feature Generation from Clinical Notes for Outcome Prediction},
  author={Wang, Jiayi and Vallon, Jacqueline Jil and Panjwani, Neil and Ling, Xi and Vij, Sushmita and Srinivas, Sandy and Leppert, John and Buyyounouski, Mark K. and Bayati, Mohsen},
  journal={arXiv preprint arXiv:2508.01956},
  year={2025},
  url={https://arxiv.org/abs/2508.01956}
}
```

---

## Questions?

For issues specific to:
- **MIMIC-IV access/structure**: See [MIMIC-IV documentation](https://mimic.mit.edu/docs/iv/)
- **This code repository**: Open an issue on GitHub
- **PhysioNet credentialing**: Contact PhysioNet support
