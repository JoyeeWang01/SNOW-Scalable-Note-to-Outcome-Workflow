-- ================================================================================
-- STEP 1: HFpEF Cohort Selection
-- ================================================================================
--
-- Purpose: Select adult patients with HFpEF (Heart Failure with preserved
--          Ejection Fraction) as primary diagnosis who survived hospitalization
--
-- Input: MIMIC-IV v3.1 tables (physionet-data)
--   - mimiciv_3_1_hosp.diagnoses_icd
--   - mimiciv_3_1_hosp.admissions
--   - mimiciv_3_1_hosp.patients
--
-- Output: Cohort of index HFpEF admissions
--   - ~2,400 unique patients (first HFpEF admission per patient)
--   - Age >= 18 years at admission
--   - Survived hospitalization (hospital_expire_flag = 0)
--
-- ICD Codes:
--   - ICD-10: I50.30, I50.31, I50.32, I50.33
--   - ICD-9:  428.30, 428.31, 428.32, 428.33
--
-- IMPORTANT: Update the table name below to match your GCP project/dataset
-- ================================================================================

-- TODO: Replace with your project and dataset name
CREATE OR REPLACE TABLE `your_project.your_dataset.hfpef_cohort` AS
WITH dx AS (
  SELECT d.subject_id, d.hadm_id, d.icd_code, d.icd_version, d.seq_num
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
  WHERE d.seq_num = 1
    AND (
      (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^I503[0-3]$')) OR
      (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code,  r'^4283[0-3]$'))
    )
),
adm AS (
  SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.insurance,
    a.language,
    a.race,
    a.hospital_expire_flag
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
),
pt AS (
  SELECT subject_id, gender, anchor_year, anchor_age
  FROM `physionet-data.mimiciv_3_1_hosp.patients`
),
adult_hfpef AS (
  SELECT
    dx.subject_id,
    dx.hadm_id,
    dx.icd_code,
    dx.icd_version,
    adm.admittime,
    adm.dischtime,
    adm.insurance,
    adm.language,
    adm.race,
    pt.gender,
    CAST(pt.anchor_age + EXTRACT(YEAR FROM adm.admittime) - pt.anchor_year AS INT64) AS age_admit,
    adm.hospital_expire_flag
  FROM dx
  JOIN adm USING (subject_id, hadm_id)
  JOIN pt  USING (subject_id)
  WHERE CAST(pt.anchor_age + EXTRACT(YEAR FROM adm.admittime) - pt.anchor_year AS INT64) >= 18
),
index_per_subject AS (
  SELECT *
  FROM adult_hfpef
  QUALIFY ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime) = 1
)
SELECT *
FROM index_per_subject
WHERE hospital_expire_flag = 0;