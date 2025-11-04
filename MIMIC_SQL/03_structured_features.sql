-- ================================================================================
-- STEP 3: Gather Baseline Structured Features and Outcomes
-- ================================================================================
--
-- Purpose: Gather structured features from MIMIC-IV for baseline comparison
--
-- Input: Table from Step 2 (cohort with discharge notes)
--
-- Output: Comprehensive feature set for each admission including:
--   - Demographics: age, gender
--   - Outcomes: 30-day mortality, 1-year mortality
--   - Vitals: heart rate, BP, SpO2, temperature, BMI (mean/min/max)
--   - Labs: bicarbonate, creatinine, hemoglobin, INR, platelets, potassium,
--           WBC, sodium, NT-proBNP, troponin
--   - Comorbidities: 15 Charlson comorbidity components (binary)
--   - Cardiovascular diagnoses: HT, CAD, PH, AF (binary)
--
-- Feature Extraction Logic:
--   - Vitals: Priority cascade ED vitals → ED triage → ICU first-day
--   - BMI: OMR measurements → ICU measurements
--   - Labs: Averaged over entire hospital admission
--   - Comorbidities: From Charlson comorbidity index (derived table)
--   - Outcomes: Calculated from patient death dates
--
-- IMPORTANT: Update the input table reference on line 31
--
-- Usage:
--   Option A (Direct export): Run query and save results as CSV
--   Option B (Save as table): Add "CREATE OR REPLACE TABLE ... AS" before WITH
-- ================================================================================

-- TODO: Update this reference to your table from Step 2
WITH coh AS (
  SELECT DISTINCT
    c.subject_id,
    c.hadm_id,
    a.admittime,
    a.dischtime,
    c.discharge_text
  FROM `your_project.your_dataset.hfpef_with_notes` c
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (subject_id, hadm_id)
),

/* ---------------- Demographics ---------------- */
demo AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.admittime,
    c.dischtime,
    c.discharge_text,                    -- carry forward for final select
    d.age AS age_admission,
    CAST(p.gender = 'M' AS INT64) AS gender
  FROM coh c
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.age` d USING (subject_id, hadm_id)
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
),

/* ---------------- Outcomes ---------------- */
outcomes AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    CAST((pt.dod IS NOT NULL) AND (DATETIME(pt.dod) > c.dischtime)
      AND (DATETIME(pt.dod) <= c.dischtime + INTERVAL 30 DAY) AS INT64) AS death_30_days,
    CAST((pt.dod IS NOT NULL) AND (DATETIME(pt.dod) > c.dischtime)
      AND (DATETIME(pt.dod) <= c.dischtime + INTERVAL 365 DAY) AS INT64) AS death_1_year
  FROM coh c
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt USING (subject_id)
),

/* ---------------- Earliest ICU & ED stays ---------------- */
first_icu AS (
  SELECT subject_id, hadm_id, stay_id, intime
  FROM (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime,
           ROW_NUMBER() OVER (PARTITION BY i.subject_id, i.hadm_id ORDER BY i.intime) rn
    FROM `physionet-data.mimiciv_3_1_icu.icustays` i
    JOIN coh USING (subject_id, hadm_id)
  ) WHERE rn = 1
),
first_ed AS (
  SELECT subject_id, hadm_id, stay_id, intime
  FROM (
    SELECT e.subject_id, e.hadm_id, e.stay_id, e.intime,
           ROW_NUMBER() OVER (PARTITION BY e.subject_id, e.hadm_id ORDER BY e.intime) rn
    FROM `physionet-data.mimiciv_ed.edstays` e
    JOIN coh USING (subject_id, hadm_id)
  ) WHERE rn = 1
),

/* ---------------- ED triage snapshot ---------------- */
ed_triage AS (
  SELECT
    fe.subject_id, fe.hadm_id,
    CASE WHEN t.temperature > 45 THEN (t.temperature - 32.0) * 5.0/9.0 ELSE t.temperature END AS ed_temperature_c,
    t.heartrate AS ed_heartrate,
    t.o2sat     AS ed_spo2,
    t.sbp       AS ed_sbp
  FROM first_ed fe
  LEFT JOIN `physionet-data.mimiciv_ed.triage` t USING (stay_id)
),

/* ---------------- ED vitalsign rows → normalized ---------------- */
ed_vs_norm AS (
  SELECT
    fe.subject_id, fe.hadm_id,
    CASE WHEN vs.temperature > 45 THEN (vs.temperature - 32.0) * 5.0/9.0 ELSE vs.temperature END AS temp_c_raw,
    vs.heartrate AS hr_raw,
    vs.sbp       AS sbp_raw,
    vs.o2sat     AS spo2_raw
  FROM first_ed fe
  LEFT JOIN `physionet-data.mimiciv_ed.vitalsign` vs USING (stay_id)
),
ed_vs_agg AS (
  SELECT
    subject_id, hadm_id,
    -- HR
    AVG(CASE WHEN hr_raw  BETWEEN 20 AND 250 THEN hr_raw END)       AS ed_hr_mean,
    MIN(CASE WHEN hr_raw  BETWEEN 20 AND 250 THEN hr_raw END)       AS ed_hr_min,
    MAX(CASE WHEN hr_raw  BETWEEN 20 AND 250 THEN hr_raw END)       AS ed_hr_max,
    -- SBP
    AVG(CASE WHEN sbp_raw BETWEEN 50 AND 260 THEN sbp_raw END)      AS ed_sbp_mean,
    MIN(CASE WHEN sbp_raw BETWEEN 50 AND 260 THEN sbp_raw END)      AS ed_sbp_min,
    MAX(CASE WHEN sbp_raw BETWEEN 50 AND 260 THEN sbp_raw END)      AS ed_sbp_max,
    -- SpO2
    AVG(CASE WHEN spo2_raw BETWEEN 50 AND 100 THEN spo2_raw END)    AS ed_spo2_mean,
    MIN(CASE WHEN spo2_raw BETWEEN 50 AND 100 THEN spo2_raw END)    AS ed_spo2_min,
    MAX(CASE WHEN spo2_raw BETWEEN 50 AND 100 THEN spo2_raw END)    AS ed_spo2_max,
    -- Temp (°C)
    AVG(CASE WHEN temp_c_raw BETWEEN 32 AND 42 THEN temp_c_raw END) AS ed_temp_mean,
    MIN(CASE WHEN temp_c_raw BETWEEN 32 AND 42 THEN temp_c_raw END) AS ed_temp_min,
    MAX(CASE WHEN temp_c_raw BETWEEN 32 AND 42 THEN temp_c_raw END) AS ed_temp_max
  FROM ed_vs_norm
  GROUP BY subject_id, hadm_id
),

/* ---------------- ICU first-day vitals ---------------- */
icu_vitals AS (
  SELECT
    c.subject_id, c.hadm_id,
    v.heart_rate_mean  AS icu_hr_mean,
    v.heart_rate_min   AS icu_hr_min,
    v.heart_rate_max   AS icu_hr_max,
    v.sbp_mean         AS icu_sbp_mean,
    v.sbp_min          AS icu_sbp_min,
    v.sbp_max          AS icu_sbp_max,
    v.spo2_mean        AS icu_spo2_mean,
    v.spo2_min         AS icu_spo2_min,
    v.spo2_max         AS icu_spo2_max,
    v.temperature_mean AS icu_temp_mean,
    v.temperature_min  AS icu_temp_min,
    v.temperature_max  AS icu_temp_max,
    SAFE_DIVIDE(w.weight, POW(h.height/100.0, 2)) AS icu_bmi
  FROM coh c
  LEFT JOIN first_icu f USING (subject_id, hadm_id)
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_vitalsign` v ON v.stay_id = f.stay_id
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_weight`      w ON w.stay_id = f.stay_id
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_height`      h ON h.stay_id = f.stay_id
),

/* ---------------- OMR nearest to admission ---------------- */
omr_candidates AS (
  SELECT
    c.subject_id, c.hadm_id, DATE(c.admittime) AS admit_date,
    o.chartdate, UPPER(o.result_name) AS result_name,
    SAFE_CAST(o.result_value AS FLOAT64) AS value
  FROM coh c
  JOIN `physionet-data.mimiciv_3_1_hosp.omr` o
    ON o.subject_id = c.subject_id
   AND o.chartdate BETWEEN DATE(c.admittime) - INTERVAL 365 DAY
                        AND DATE(c.admittime) + INTERVAL 7 DAY
),
omr_weight AS (
  SELECT subject_id, hadm_id, value AS omr_weight_kg_raw
  FROM (
    SELECT subject_id, hadm_id, value,
           ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id
                              ORDER BY ABS(DATE_DIFF(chartdate, admit_date, DAY))) rn
    FROM omr_candidates
    WHERE result_name IN ('WEIGHT (KG)','WEIGHT')
  ) WHERE rn = 1
),
omr_height AS (
  SELECT subject_id, hadm_id, value AS omr_height_cm_raw
  FROM (
    SELECT subject_id, hadm_id, value,
           ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id
                              ORDER BY ABS(DATE_DIFF(chartdate, admit_date, DAY))) rn
    FROM omr_candidates
    WHERE result_name IN ('HEIGHT (CM)','HEIGHT')
  ) WHERE rn = 1
),
omr_bmi AS (
  SELECT
    c.subject_id, c.hadm_id,
    CASE
      WHEN w.omr_weight_kg_raw BETWEEN 30 AND 300
       AND h.omr_height_cm_raw  BETWEEN 120 AND 220
      THEN SAFE_DIVIDE(w.omr_weight_kg_raw, POW(h.omr_height_cm_raw/100.0, 2))
      ELSE NULL
    END AS omr_bmi
  FROM coh c
  LEFT JOIN omr_weight w USING (subject_id, hadm_id)
  LEFT JOIN omr_height h USING (subject_id, hadm_id)
),

/* ---------------- LAB aggregates over entire admission ---------------- */
chem_all AS (
  SELECT ch.hadm_id, AVG(ch.creatinine) AS creatinine
  FROM `physionet-data.mimiciv_3_1_derived.chemistry` ch
  GROUP BY ch.hadm_id
),
bg_all AS (
  SELECT bg.hadm_id,
         AVG(bg.bicarbonate) AS bicarbonate,
         AVG(bg.sodium)      AS sodium,
         AVG(bg.potassium)   AS potassium
  FROM `physionet-data.mimiciv_3_1_derived.bg` bg
  GROUP BY bg.hadm_id
),
cbc_all AS (
  SELECT cbc.hadm_id,
         AVG(cbc.hemoglobin) AS hemoglobin,
         AVG(cbc.platelet)   AS platelet_count,
         AVG(cbc.wbc)        AS wbc_count
  FROM `physionet-data.mimiciv_3_1_derived.complete_blood_count` cbc
  GROUP BY cbc.hadm_id
),
coag_all AS (
  SELECT cg.hadm_id, AVG(cg.inr) AS inr
  FROM `physionet-data.mimiciv_3_1_derived.coagulation` cg
  GROUP BY cg.hadm_id
),
cm_all AS (
  SELECT cm.hadm_id,
         MAX(cm.ntprobnp)   AS ntprobnp,
         MAX(cm.troponin_t) AS troponin
  FROM `physionet-data.mimiciv_3_1_derived.cardiac_marker` cm
  GROUP BY cm.hadm_id
),
labs AS (
  SELECT
    d.subject_id, d.hadm_id,
    chem.creatinine,
    bg.bicarbonate, bg.sodium, bg.potassium,
    cbc.hemoglobin, cbc.platelet_count, cbc.wbc_count,
    coag.inr,
    cm.ntprobnp, cm.troponin
  FROM demo d
  LEFT JOIN chem_all chem ON chem.hadm_id = d.hadm_id
  LEFT JOIN bg_all   bg   ON bg.hadm_id   = d.hadm_id
  LEFT JOIN cbc_all  cbc  ON cbc.hadm_id  = d.hadm_id
  LEFT JOIN coag_all coag ON coag.hadm_id = d.hadm_id
  LEFT JOIN cm_all   cm   ON cm.hadm_id   = d.hadm_id
),

/* ---------------- Charlson (per admission) ---------------- */
charlson AS (
  SELECT
    c.hadm_id,
    CAST(myocardial_infarct          AS INT64) AS acute_myocardial_infarction,
    CAST(peripheral_vascular_disease AS INT64) AS peripheral_vascular_disease,
    CAST(cerebrovascular_disease     AS INT64) AS cerebrovascular_disease,
    CAST(dementia                    AS INT64) AS dementia,
    CAST(chronic_pulmonary_disease   AS INT64) AS chronic_obstructive_pulmonary_disease,
    CAST(rheumatic_disease           AS INT64) AS rheumatoid_disease,
    CAST(peptic_ulcer_disease        AS INT64) AS peptic_ulcer_disease,
    CAST(mild_liver_disease          AS INT64) AS mild_liver_disease,
    CAST(diabetes_without_cc         AS INT64) AS diabetes,
    CAST(diabetes_with_cc            AS INT64) AS diabetes_complications,
    CAST(paraplegia                  AS INT64) AS hemiplegia_paraplegia,
    CAST(renal_disease               AS INT64) AS renal_disease,
    CAST(malignant_cancer            AS INT64) AS cancer,
    CAST(severe_liver_disease        AS INT64) AS severe_liver_disease,
    CAST(malignant_cancer            AS INT64) AS malignant_cancer
  FROM `physionet-data.mimiciv_3_1_derived.charlson` ch
  JOIN coh c USING (hadm_id)
),

/* ---------------- Dx-based cardiovascular flags ---------------- */
extras AS (
  SELECT
    c.subject_id, c.hadm_id,
    MAX(IF(
      (d.icd_version = 10 AND SUBSTR(d.icd_code,1,3) IN ('I10','I11','I12','I13','I15','I16'))
      OR (d.icd_version = 9  AND (STARTS_WITH(d.icd_code,'401') OR STARTS_WITH(d.icd_code,'402')
                               OR STARTS_WITH(d.icd_code,'403') OR STARTS_WITH(d.icd_code,'404')
                               OR STARTS_WITH(d.icd_code,'405'))), 1, 0)) AS HT,
    MAX(IF((d.icd_version = 10 AND STARTS_WITH(d.icd_code,'I25'))
        OR (d.icd_version = 9  AND STARTS_WITH(d.icd_code,'414')), 1, 0)) AS CAD,
    MAX(IF((d.icd_version = 10 AND STARTS_WITH(d.icd_code,'I27'))
        OR (d.icd_version = 9  AND STARTS_WITH(d.icd_code,'416')), 1, 0)) AS PH,
    MAX(IF((d.icd_version = 10 AND STARTS_WITH(d.icd_code,'I48'))
        OR (d.icd_version = 9  AND (STARTS_WITH(d.icd_code,'42731') OR STARTS_WITH(d.icd_code,'42732'))), 1, 0)) AS AF
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
  JOIN coh c USING (subject_id, hadm_id)
  GROUP BY c.subject_id, c.hadm_id
),

/* ---------------- Fuse vitals: ED → ED triage → ICU; BMI: OMR → ICU ---------------- */
vitals_fused AS (
  SELECT
    d.subject_id, d.hadm_id,

    -- Heart rate
    COALESCE(ea.ed_hr_mean,  et.ed_heartrate,  iv.icu_hr_mean) AS heart_rate_mean,
    COALESCE(ea.ed_hr_min,   et.ed_heartrate,  iv.icu_hr_min)  AS heart_rate_min,
    COALESCE(ea.ed_hr_max,   et.ed_heartrate,  iv.icu_hr_max)  AS heart_rate_max,

    -- Systolic BP
    COALESCE(ea.ed_sbp_mean, et.ed_sbp, iv.icu_sbp_mean) AS systolic_bp_mean,
    COALESCE(ea.ed_sbp_min,  et.ed_sbp, iv.icu_sbp_min)  AS systolic_bp_min,
    COALESCE(ea.ed_sbp_max,  et.ed_sbp, iv.icu_sbp_max)  AS systolic_bp_max,

    -- Oxygen saturation
    COALESCE(ea.ed_spo2_mean, et.ed_spo2, iv.icu_spo2_mean) AS oxygen_saturation_mean,
    COALESCE(ea.ed_spo2_min,  et.ed_spo2, iv.icu_spo2_min)  AS oxygen_saturation_min,
    COALESCE(ea.ed_spo2_max,  et.ed_spo2, iv.icu_spo2_max)  AS oxygen_saturation_max,

    -- Temperature (°C)
    COALESCE(ea.ed_temp_mean, et.ed_temperature_c, iv.icu_temp_mean) AS temperature_mean,
    COALESCE(ea.ed_temp_min,  et.ed_temperature_c, iv.icu_temp_min)  AS temperature_min,
    COALESCE(ea.ed_temp_max,  et.ed_temperature_c, iv.icu_temp_max)  AS temperature_max,

    -- BMI: prefer OMR, else ICU
    COALESCE(ob.omr_bmi, iv.icu_bmi) AS bmi

  FROM demo d
  LEFT JOIN ed_triage  et USING (subject_id, hadm_id)
  LEFT JOIN ed_vs_agg  ea USING (subject_id, hadm_id)
  LEFT JOIN icu_vitals iv USING (subject_id, hadm_id)
  LEFT JOIN omr_bmi    ob USING (subject_id, hadm_id)
)

/* ---------------- Final feature table (includes discharge_text) ---------------- */
SELECT
  d.subject_id,
  d.hadm_id,

  -- targets
  o.death_30_days,
  o.death_1_year,

  -- demographics
  d.age_admission,
  d.gender,

  -- notes
  d.discharge_text,

  -- vitals (ED→ED triage→ICU fused; mean/min/max only)
  v.heart_rate_mean,        v.heart_rate_min,        v.heart_rate_max,
  v.systolic_bp_mean,       v.systolic_bp_min,       v.systolic_bp_max,
  v.oxygen_saturation_mean, v.oxygen_saturation_min, v.oxygen_saturation_max,
  v.temperature_mean,       v.temperature_min,       v.temperature_max,
  v.bmi,

  -- labs (aggregated over entire admission)
  l.bicarbonate,
  l.creatinine,
  l.hemoglobin,
  l.inr,
  l.platelet_count,
  l.potassium,
  l.wbc_count,
  l.sodium,
  l.ntprobnp,
  l.troponin,

  -- comorbidities
  c.acute_myocardial_infarction,
  c.peripheral_vascular_disease,
  c.cerebrovascular_disease,
  c.dementia,
  c.chronic_obstructive_pulmonary_disease,
  c.rheumatoid_disease,
  c.peptic_ulcer_disease,
  c.mild_liver_disease,
  c.diabetes,
  c.diabetes_complications,
  c.hemiplegia_paraplegia,
  c.renal_disease,
  c.cancer,
  c.severe_liver_disease,
  c.malignant_cancer,

  -- dx flags
  e.HT, e.CAD, e.PH, e.AF

FROM demo d
LEFT JOIN outcomes     o USING (subject_id, hadm_id)
LEFT JOIN vitals_fused v USING (subject_id, hadm_id)
LEFT JOIN labs         l USING (subject_id, hadm_id)
LEFT JOIN charlson     c USING (hadm_id)
LEFT JOIN extras       e USING (subject_id, hadm_id);