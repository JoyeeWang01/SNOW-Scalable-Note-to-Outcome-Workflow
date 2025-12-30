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
Extracts comprehensive structured baseline features for each patient admission with intelligent data fusion across multiple sources (ED, ICU, OMR) and quality filters.

**Total Features:** 38 structured features + 2 outcomes + 2 identifiers = 42 columns

#### Feature Categories:

**1. OUTCOMES (2 features)**
- `death_30_days` - Binary indicator for death within 30 days post-discharge
- `death_1_year` - Binary indicator for death within 1 year post-discharge

**2. DEMOGRAPHICS (2 features)**
- `age_admission` - Age at admission in years
- `gender` - Binary (1=Male, 0=Female)

**3. VITAL SIGNS (5 features)** - Prioritized fusion: ED aggregated → ED triage → ICU first-day
- `temperature` - Body temperature in Celsius (auto-converts F to C when >45°)
- `heart_rate` - Heart rate in beats per minute (range: 20-250)
- `oxygen_saturation` - SpO2 percentage (range: 50-100%)
- `systolic_bp` - Systolic blood pressure in mmHg (range: 50-260)
- `bmi` - Body mass index in kg/m² (Prioritized: OMR → ICU, range: 10-80)

**4. LABORATORY VALUES (10 features)** - Derived tables prioritized over raw labevents
- `bicarbonate` - Bicarbonate in mEq/L (AVG)
- `creatinine` - Creatinine in mg/dL (AVG)
- `hemoglobin` - Hemoglobin in g/dL (AVG)
- `inr` - International Normalized Ratio (AVG)
- `platelet_count` - Platelet count in K/uL (AVG)
- `potassium` - Potassium in mEq/L (AVG)
- `wbc_count` - White blood cell count in K/uL (AVG)
- `sodium` - Sodium in mEq/L (AVG)
- `ntprobnp` - NT-proBNP in pg/mL (MAX - peak value)
- `troponin` - Troponin T (MAX - peak value)

**5. COMORBIDITIES (15 features)** - Binary indicators from Charlson comorbidity index
- `acute_myocardial_infarction`
- `peripheral_vascular_disease`
- `cerebrovascular_disease`
- `dementia`
- `chronic_obstructive_pulmonary_disease`
- `rheumatoid_disease`
- `peptic_ulcer_disease`
- `mild_liver_disease`
- `diabetes`
- `diabetes_complications`
- `hemiplegia_paraplegia`
- `renal_disease`
- `cancer`
- `moderate_severe_liver_disease`
- `metastatic_cancer`

**6. DIAGNOSIS FLAGS (4 features)** - ICD-9/10 based binary indicators
- `HT` - Hypertension (ICD-10: I10-I16; ICD-9: 401-405)
- `CAD` - Coronary artery disease (ICD-10: I25; ICD-9: 414)
- `PH` - Pulmonary hypertension (ICD-10: I27; ICD-9: 416)
- `AF` - Atrial fibrillation (ICD-10: I48; ICD-9: 42731, 42732)

#### Data Quality Features:
- **Plausibility filters** on all vitals and anthropometric values
- **Temperature auto-conversion** from Fahrenheit to Celsius
- **Unit conversions** for OMR weight (lbs→kg) and height (inches→cm)
- **Prioritized data fusion** across ED, ICU, and OMR sources
- **COALESCE strategy** for labs: derived concept tables preferred over raw labevents
- **Peak values** (MAX) for cardiac biomarkers (NTproBNP, Troponin)
- **Average values** (AVG) for all other labs and vitals

**Prerequisites:** Step 2 completed

**Output:** Final feature table with 42 columns

**To run:**
1. **Update the cohort table reference** (line ~139 in the SQL file):
   ```sql
   FROM `your_project.your_dataset.hfpef_with_notes`
   ```

2. **Choose an execution method:**

   **Option A - Export results directly to CSV:**
   - Run the query in BigQuery Console
   - Click "Save Results" → "CSV (local file)"
   - Save as `data/structured_features.csv`

   **Option B - Save as a BigQuery table first:**
   - Add at the beginning of the SQL file:
     ```sql
     CREATE OR REPLACE TABLE `your_project.your_dataset.structured_features` AS
     ```
   - Run the query
   - Then export using `bq` command (see Data Export section below)

3. **(Optional) Customize plausibility ranges** if needed:
   - HR: 20-250 bpm (line ~237)
   - SBP: 50-260 mmHg (line ~239)
   - SpO2: 50-100% (line ~241)
   - Temperature: 32-42°C (line ~243)
   - BMI: 10-80 kg/m² (line ~379)
   - Weight: 30-300 kg (line ~259, 380)
   - Height: 120-220 cm (line ~260, 381)

**Expected result:** Same number of rows as Step 2 (~2,400), with 42 columns total

**Important notes:**
- Missing values are represented as NULL (no imputation)
- For patients with multiple ICU/ED stays, only the FIRST stay is used
- OMR values are selected from ±365 days around admission (closest by date)
- See the comprehensive header comments in `03_structured_features.sql` for full documentation

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
| 3. Structured Features | Final dataset | ~2,400 | 42 | 38 features + 2 outcomes + 2 IDs |

**Note:** Step 3 includes comprehensive data quality features:
- Intelligent data fusion across ED, ICU, and OMR sources
- Automatic unit conversions (temperature, weight, height)
- Plausibility filters on all vital signs and anthropometric measurements
- Prioritized COALESCE strategy for lab values (derived tables → raw labevents)

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
