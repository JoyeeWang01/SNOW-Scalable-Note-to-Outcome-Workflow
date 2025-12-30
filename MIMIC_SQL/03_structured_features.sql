-- ================================================================================
-- STRUCTURED FEATURES EXTRACTION FOR HFPEF COHORT
-- ================================================================================
--
-- PURPOSE:
--   Extract structured baseline features for heart failure with preserved ejection
--   fraction (HFpEF) patients from MIMIC-IV v3.1. Creates a single row per hospital
--   admission (hadm_id) with demographics, vitals, labs, and comorbidities.
--
-- INPUT:
--   Requires output from 02_hfpef_cohort.sql:
--   - `your_project.your_dataset.hfpef_with_notes` (cohort table)
--
-- OUTPUT:
--   Feature table with the following categories:
--   1. OUTCOMES (2 features)
--      - death_30_days: Binary indicator for death within 30 days post-discharge
--      - death_1_year: Binary indicator for death within 1 year post-discharge
--
--   2. DEMOGRAPHICS (2 features)
--      - age_admission: Age at admission in years
--      - gender: Binary (1=Male, 0=Female)
--
--   3. VITALS (5 features) - Prioritized fusion: ED aggregated → ED triage → ICU first-day
--      - temperature: Body temperature in Celsius (auto-converts F to C)
--      - heart_rate: Heart rate in beats per minute
--      - oxygen_saturation: SpO2 percentage (50-100%)
--      - systolic_bp: Systolic blood pressure in mmHg
--      - bmi: Body mass index (kg/m²) - Prioritized: OMR → ICU
--
--   4. LABORATORY VALUES (10 features) - Derived tables prioritized over raw labevents
--      - bicarbonate: Bicarbonate in mEq/L (AVG)
--      - creatinine: Creatinine in mg/dL (AVG)
--      - hemoglobin: Hemoglobin in g/dL (AVG)
--      - inr: International Normalized Ratio (AVG)
--      - platelet_count: Platelet count in K/uL (AVG)
--      - potassium: Potassium in mEq/L (AVG)
--      - wbc_count: White blood cell count in K/uL (AVG)
--      - sodium: Sodium in mEq/L (AVG)
--      - ntprobnp: NT-proBNP in pg/mL (MAX - peak value)
--      - troponin: Troponin T (MAX - peak value)
--
--   5. COMORBIDITIES (15 features) - Binary indicators from Charlson comorbidity index
--      - acute_myocardial_infarction
--      - peripheral_vascular_disease
--      - cerebrovascular_disease
--      - dementia
--      - chronic_obstructive_pulmonary_disease
--      - rheumatoid_disease
--      - peptic_ulcer_disease
--      - mild_liver_disease
--      - diabetes
--      - diabetes_complications
--      - hemiplegia_paraplegia
--      - renal_disease
--      - cancer
--      - moderate_severe_liver_disease
--      - metastatic_cancer
--
--   6. DIAGNOSIS FLAGS (4 features) - ICD-9/10 based binary indicators
--      - HT: Hypertension (ICD-10: I10-I16; ICD-9: 401-405)
--      - CAD: Coronary artery disease (ICD-10: I25; ICD-9: 414)
--      - PH: Pulmonary hypertension (ICD-10: I27; ICD-9: 416)
--      - AF: Atrial fibrillation (ICD-10: I48; ICD-9: 42731, 42732)
--
-- TOTAL: 38 structured features + 2 outcomes + 2 identifiers (subject_id, hadm_id)
--
-- DATA QUALITY FEATURES:
--   - Plausibility filters on vitals and anthropometric values
--   - Temperature auto-conversion (F to C) when values exceed 45°C
--   - Unit conversions for OMR weight (lbs→kg) and height (inches→cm)
--   - Prioritized data fusion across ED, ICU, and OMR sources
--   - COALESCE strategy: derived concept tables preferred over raw labevents
--
-- MIMIC-IV TABLES USED:
--   - physionet-data.mimiciv_3_1_hosp.patients (demographics, death dates)
--   - physionet-data.mimiciv_3_1_hosp.diagnoses_icd (diagnosis codes)
--   - physionet-data.mimiciv_3_1_hosp.labevents (raw lab values)
--   - physionet-data.mimiciv_3_1_hosp.omr (outpatient measurements)
--   - physionet-data.mimiciv_3_1_icu.icustays (ICU admission times)
--   - physionet-data.mimiciv_3_1_ed.edstays (ED admission times)
--   - physionet-data.mimiciv_3_1_ed.triage (ED triage vitals)
--   - physionet-data.mimiciv_3_1_ed.vitalsign (ED vital signs time series)
--   - physionet-data.mimiciv_3_1_derived.age (age at admission)
--   - physionet-data.mimiciv_3_1_derived.charlson (Charlson comorbidity scores)
--   - physionet-data.mimiciv_3_1_derived.first_day_vitalsign (ICU first-day vitals)
--   - physionet-data.mimiciv_3_1_derived.first_day_weight (ICU first-day weight)
--   - physionet-data.mimiciv_3_1_derived.first_day_height (ICU first-day height)
--   - physionet-data.mimiciv_3_1_derived.chemistry (derived chemistry labs)
--   - physionet-data.mimiciv_3_1_derived.bg (blood gas values)
--   - physionet-data.mimiciv_3_1_derived.complete_blood_count (CBC values)
--   - physionet-data.mimiciv_3_1_derived.coagulation (coagulation values)
--   - physionet-data.mimiciv_3_1_derived.cardiac_marker (cardiac biomarkers)
--
-- CUSTOMIZATION INSTRUCTIONS:
--   1. Update the cohort table reference in the 'coh' CTE (line ~13):
--      Change `your_project.your_dataset.hfpef_with_notes` to your actual table path
--
--   2. Adjust lab itemids if using different MIMIC-IV versions (see lines 276-297)
--
--   3. Modify plausibility ranges if needed:
--      - HR: 20-250 bpm (line 111)
--      - SBP: 50-260 mmHg (line 113)
--      - SpO2: 50-100% (line 115)
--      - Temperature: 32-42°C (line 117)
--      - BMI: 10-80 kg/m² (line 253)
--      - Weight: 30-300 kg (line 133, 254)
--      - Height: 120-220 cm (line 134, 255)
--
--   4. Modify outcome windows:
--      - 30-day mortality: line 38
--      - 1-year mortality: line 44
--
-- USAGE:
--   Run this query in BigQuery and save results to your dataset:
--
--   ```sql
--   CREATE OR REPLACE TABLE `your_project.your_dataset.structured_features` AS
--   -- [Paste this entire query]
--   ```
--
-- NOTES:
--   - Missing values are represented as NULL (no imputation)
--   - For patients with multiple ICU/ED stays, only the FIRST stay is used
--   - OMR values are selected from ±365 days around admission (closest by date)
--   - Peak values (MAX) are used for cardiac biomarkers (NTproBNP, Troponin)
--   - Average values (AVG) are used for all other labs and vitals
--
-- LAST MODIFIED: 2025
-- ================================================================================

WITH coh AS (
  -- Base cohort: HFpEF patients with notes (already filtered)
  SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime
  FROM `your_project.your_dataset.hfpef_with_notes`
),

/* ---------------- Demographics ---------------- */
demo AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.admittime,
    c.dischtime,
    d.age AS age_admission,
    CAST(p.gender = 'M' AS INT64) AS gender
  FROM coh c
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.age`   d USING (subject_id, hadm_id)
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
),

/* ---------------- Outcomes: death within 30 days / 1 year after discharge ---------------- */
outcomes AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    CAST(
      pt.dod IS NOT NULL
      AND DATETIME(pt.dod) > c.dischtime
      AND DATETIME(pt.dod) <= c.dischtime + INTERVAL 30 DAY
      AS INT64
    ) AS death_30_days,
    CAST(
      pt.dod IS NOT NULL
      AND DATETIME(pt.dod) > c.dischtime
      AND DATETIME(pt.dod) <= c.dischtime + INTERVAL 365 DAY
      AS INT64
    ) AS death_1_year
  FROM coh c
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt USING (subject_id)
),

/* ---------------- Earliest ICU & ED stays ---------------- */
first_icu AS (
  SELECT subject_id, hadm_id, stay_id, intime
  FROM (
    SELECT
      i.subject_id, i.hadm_id, i.stay_id, i.intime,
      ROW_NUMBER() OVER (PARTITION BY i.subject_id, i.hadm_id ORDER BY i.intime) AS rn
    FROM `physionet-data.mimiciv_3_1_icu.icustays` i
    JOIN coh USING (subject_id, hadm_id)
  )
  WHERE rn = 1
),
first_ed AS (
  SELECT subject_id, hadm_id, stay_id, intime
  FROM (
    SELECT
      e.subject_id, e.hadm_id, e.stay_id, e.intime,
      ROW_NUMBER() OVER (PARTITION BY e.subject_id, e.hadm_id ORDER BY e.intime) AS rn
    FROM `physionet-data.mimiciv_ed.edstays` e
    JOIN coh USING (subject_id, hadm_id)
  )
  WHERE rn = 1
),

/* ---------------- ED triage snapshot ---------------- */
ed_triage AS (
  SELECT
    fe.subject_id,
    fe.hadm_id,
    CASE
      WHEN t.temperature > 45 THEN (t.temperature - 32.0) * 5.0/9.0
      ELSE t.temperature
    END AS ed_temperature_c,
    t.heartrate AS ed_heartrate,
    t.o2sat     AS ed_spo2,
    t.sbp       AS ed_sbp
  FROM first_ed fe
  LEFT JOIN `physionet-data.mimiciv_ed.triage` t USING (stay_id)
),

/* ---------------- ED vitalsign rows → normalized ---------------- */
ed_vs_norm AS (
  SELECT
    fe.subject_id,
    fe.hadm_id,
    CASE
      WHEN vs.temperature > 45 THEN (vs.temperature - 32.0) * 5.0/9.0
      ELSE vs.temperature
    END AS temp_c_raw,
    vs.heartrate AS hr_raw,
    vs.sbp       AS sbp_raw,
    vs.o2sat     AS spo2_raw
  FROM first_ed fe
  LEFT JOIN `physionet-data.mimiciv_ed.vitalsign` vs USING (stay_id)
),
ed_vs_agg AS (
  SELECT
    subject_id,
    hadm_id,
    -- HR
    AVG(CASE WHEN hr_raw  BETWEEN 20 AND 250 THEN hr_raw END)       AS ed_hr_mean,
    -- SBP
    AVG(CASE WHEN sbp_raw BETWEEN 50 AND 260 THEN sbp_raw END)      AS ed_sbp_mean,
    -- SpO2
    AVG(CASE WHEN spo2_raw BETWEEN 50 AND 100 THEN spo2_raw END)    AS ed_spo2_mean,
    -- Temp (°C)
    AVG(CASE WHEN temp_c_raw BETWEEN 32 AND 42 THEN temp_c_raw END) AS ed_temp_mean
  FROM ed_vs_norm
  GROUP BY subject_id, hadm_id
),

/* ---------------- ICU first-day vitals + ICU BMI ---------------- */
icu_vitals AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    v.heart_rate_mean  AS icu_hr_mean,
    v.sbp_mean         AS icu_sbp_mean,
    v.spo2_mean        AS icu_spo2_mean,
    v.temperature_mean AS icu_temp_mean,
    -- ICU BMI with basic plausibility checks
    CASE
      WHEN w.weight BETWEEN 30 AND 300
       AND h.height BETWEEN 120 AND 220
      THEN SAFE_DIVIDE(w.weight, POW(h.height/100.0, 2))
      ELSE NULL
    END AS icu_bmi
  FROM coh c
  LEFT JOIN first_icu f USING (subject_id, hadm_id)
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_vitalsign` v
    ON v.stay_id = f.stay_id
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_weight` w
    ON w.stay_id = f.stay_id
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_height` h
    ON h.stay_id = f.stay_id
),

/* ---------------- OMR nearest to admission (for BMI) ---------------- */
omr_candidates AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    DATE(c.admittime) AS admit_date,
    o.chartdate,
    UPPER(o.result_name) AS result_name,
    SAFE_CAST(o.result_value AS FLOAT64) AS value
  FROM coh c
  JOIN `physionet-data.mimiciv_3_1_hosp.omr` o
    ON o.subject_id = c.subject_id
   AND o.chartdate BETWEEN DATE(c.admittime) - INTERVAL 365 DAY
                        AND DATE(c.admittime) + INTERVAL 7 DAY
),

-- OMR weight (kg), allowing both kg and lbs
omr_weight AS (
  SELECT
    subject_id,
    hadm_id,
    CASE
      WHEN result_name IN ('WEIGHT (KG)', 'WEIGHT')
        THEN value
      WHEN result_name IN ('WEIGHT (LBS)', 'WEIGHT (LB)', 'WEIGHT (POUNDS)')
        THEN value * 0.453592
      ELSE NULL
    END AS omr_weight_kg
  FROM (
    SELECT
      subject_id,
      hadm_id,
      result_name,
      value,
      ROW_NUMBER() OVER (
        PARTITION BY subject_id, hadm_id
        ORDER BY ABS(DATE_DIFF(chartdate, admit_date, DAY))
      ) AS rn
    FROM omr_candidates
    WHERE result_name IN (
      'WEIGHT (KG)', 'WEIGHT',
      'WEIGHT (LBS)', 'WEIGHT (LB)', 'WEIGHT (POUNDS)'
    )
  )
  WHERE rn = 1
),

-- OMR height (cm), allowing both cm and inches
omr_height AS (
  SELECT
    subject_id,
    hadm_id,
    CASE
      WHEN result_name IN ('HEIGHT (CM)', 'HEIGHT')
        THEN value
      WHEN result_name IN ('HEIGHT (INCHES)', 'HEIGHT (IN)')
        THEN value * 2.54
      ELSE NULL
    END AS omr_height_cm
  FROM (
    SELECT
      subject_id,
      hadm_id,
      result_name,
      value,
      ROW_NUMBER() OVER (
        PARTITION BY subject_id, hadm_id
        ORDER BY ABS(DATE_DIFF(chartdate, admit_date, DAY))
      ) AS rn
    FROM omr_candidates
    WHERE result_name IN (
      'HEIGHT (CM)', 'HEIGHT',
      'HEIGHT (INCHES)', 'HEIGHT (IN)'
    )
  )
  WHERE rn = 1
),

-- Direct OMR BMI (BMI (kg/m2))
omr_bmi_direct AS (
  SELECT
    subject_id,
    hadm_id,
    value AS omr_bmi_direct
  FROM (
    SELECT
      subject_id,
      hadm_id,
      value,
      ROW_NUMBER() OVER (
        PARTITION BY subject_id, hadm_id
        ORDER BY ABS(DATE_DIFF(chartdate, admit_date, DAY))
      ) AS rn
    FROM omr_candidates
    WHERE result_name IN ('BMI (KG/M2)', 'BMI')
  )
  WHERE rn = 1
),

-- Final OMR BMI: prefer direct OMR BMI, else from weight/height
omr_bmi AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    CASE
      WHEN bd.omr_bmi_direct BETWEEN 10 AND 80 THEN bd.omr_bmi_direct
      WHEN w.omr_weight_kg BETWEEN 30 AND 300
       AND h.omr_height_cm BETWEEN 120 AND 220
      THEN SAFE_DIVIDE(w.omr_weight_kg, POW(h.omr_height_cm/100.0, 2))
      ELSE NULL
    END AS omr_bmi
  FROM coh c
  LEFT JOIN omr_weight     w  USING (subject_id, hadm_id)
  LEFT JOIN omr_height     h  USING (subject_id, hadm_id)
  LEFT JOIN omr_bmi_direct bd USING (subject_id, hadm_id)
),

/* ---------------- LAB aggregates over entire admission (RAW from labevents, using your itemids) ---------------- */
lab_events AS (
  SELECT
    le.subject_id,
    le.hadm_id,
    le.itemid,
    le.valuenum
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
  JOIN coh c
    USING (subject_id, hadm_id)
  WHERE le.valuenum IS NOT NULL
    AND le.itemid IN (
      -- Bicarbonate
      52039, 50803, 50882,
      -- Creatinine
      52024, 50912, 52546,
      -- Sodium
      50824, 52455, 50983, 52623,
      -- Potassium
      50822, 52452, 50971, 52610,
      -- Hemoglobin
      50811, 51640, 51222,
      -- Platelet Count
      53189,
      -- WBC (White Blood Cells)
      51755, 51756,
      -- INR(PT)
      51237, 51675,
      -- NTproBNP
      50963,
      -- Troponin T
      51003
    )
),
labs_raw AS (
  SELECT
    subject_id,
    hadm_id,

    -- Bicarbonate (mEq/L)
    AVG(CASE
          WHEN itemid IN (52039, 50803, 50882)
          THEN valuenum
        END) AS bicarbonate,

    -- Creatinine (mg/dL)
    AVG(CASE
          WHEN itemid IN (52024, 50912, 52546)
          THEN valuenum
        END) AS creatinine,

    -- Sodium (mEq/L)
    AVG(CASE
          WHEN itemid IN (50824, 52455, 50983, 52623)
          THEN valuenum
        END) AS sodium,

    -- Potassium (mEq/L)
    AVG(CASE
          WHEN itemid IN (50822, 52452, 50971, 52610)
          THEN valuenum
        END) AS potassium,

    -- Hemoglobin (g/dL)
    AVG(CASE
          WHEN itemid IN (50811, 51640, 51222)
          THEN valuenum
        END) AS hemoglobin,

    -- Platelet Count (K/uL)
    AVG(CASE
          WHEN itemid = 53189
          THEN valuenum
        END) AS platelet_count,

    -- WBC Count (K/uL)
    AVG(CASE
          WHEN itemid IN (51755, 51756)
          THEN valuenum
        END) AS wbc_count,

    -- INR
    AVG(CASE
          WHEN itemid IN (51237, 51675)
          THEN valuenum
        END) AS inr,

    -- NTproBNP (pg/mL) – peak
    MAX(CASE
          WHEN itemid = 50963
          THEN valuenum
        END) AS ntprobnp,

    -- Troponin T – peak
    MAX(CASE
          WHEN itemid = 51003
          THEN valuenum
        END) AS troponin

  FROM lab_events
  GROUP BY subject_id, hadm_id
),

/* ---------------- LAB aggregates from DERIVED concept tables ---------------- */
chem_derived AS (
  SELECT
    hadm_id,
    AVG(creatinine) AS creatinine_chem
  FROM `physionet-data.mimiciv_3_1_derived.chemistry`
  GROUP BY hadm_id
),
bg_derived AS (
  SELECT
    hadm_id,
    AVG(bicarbonate) AS bicarbonate_bg,
    AVG(sodium)      AS sodium_bg,
    AVG(potassium)   AS potassium_bg
  FROM `physionet-data.mimiciv_3_1_derived.bg`
  GROUP BY hadm_id
),
cbc_derived AS (
  SELECT
    hadm_id,
    AVG(hemoglobin) AS hemoglobin_cbc,
    AVG(platelet)   AS platelet_cbc,
    AVG(wbc)        AS wbc_cbc
  FROM `physionet-data.mimiciv_3_1_derived.complete_blood_count`
  GROUP BY hadm_id
),
coag_derived AS (
  SELECT
    hadm_id,
    AVG(inr) AS inr_coag
  FROM `physionet-data.mimiciv_3_1_derived.coagulation`
  GROUP BY hadm_id
),
cm_derived AS (
  SELECT
    hadm_id,
    MAX(ntprobnp)   AS ntprobnp_cm,
    MAX(troponin_t) AS troponin_cm
  FROM `physionet-data.mimiciv_3_1_derived.cardiac_marker`
  GROUP BY hadm_id
),

/* ---------------- Final LABS = COALESCE(derived, raw) ---------------- */
labs AS (
  SELECT
    c.subject_id,
    c.hadm_id,

    COALESCE(bg.bicarbonate_bg,        lr.bicarbonate)   AS bicarbonate,
    COALESCE(chem.creatinine_chem,     lr.creatinine)    AS creatinine,
    COALESCE(cbc.hemoglobin_cbc,       lr.hemoglobin)    AS hemoglobin,
    COALESCE(coag.inr_coag,            lr.inr)           AS inr,
    COALESCE(cbc.platelet_cbc,         lr.platelet_count) AS platelet_count,
    COALESCE(cbc.wbc_cbc,              lr.wbc_count)     AS wbc_count,
    COALESCE(bg.sodium_bg,             lr.sodium)        AS sodium,
    COALESCE(bg.potassium_bg,          lr.potassium)     AS potassium,
    COALESCE(cm.ntprobnp_cm,           lr.ntprobnp)      AS ntprobnp,
    COALESCE(cm.troponin_cm,           lr.troponin)      AS troponin

  FROM coh c
  LEFT JOIN labs_raw    lr   USING (subject_id, hadm_id)
  LEFT JOIN chem_derived chem USING (hadm_id)
  LEFT JOIN bg_derived   bg   USING (hadm_id)
  LEFT JOIN cbc_derived  cbc  USING (hadm_id)
  LEFT JOIN coag_derived coag USING (hadm_id)
  LEFT JOIN cm_derived   cm   USING (hadm_id)
),

/* ---------------- Charlson comorbidity components ---------------- */
charlson AS (
  SELECT
    hadm_id,
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
    CAST(severe_liver_disease        AS INT64) AS moderate_severe_liver_disease,
    CAST(metastatic_solid_tumor      AS INT64) AS metastatic_cancer
  FROM `physionet-data.mimiciv_3_1_derived.charlson`
),

/* ---------------- Dx-based cardiovascular flags (HT, CAD, PH, AF) ---------------- */
extras AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    MAX(
      IF(
        (d.icd_version = 10 AND SUBSTR(d.icd_code,1,3) IN ('I10','I11','I12','I13','I15','I16'))
        OR (d.icd_version = 9  AND (
              STARTS_WITH(d.icd_code,'401') OR STARTS_WITH(d.icd_code,'402') OR
              STARTS_WITH(d.icd_code,'403') OR STARTS_WITH(d.icd_code,'404') OR
              STARTS_WITH(d.icd_code,'405')
            )),
        1, 0)
    ) AS HT,
    MAX(
      IF(
        (d.icd_version = 10 AND STARTS_WITH(d.icd_code,'I25'))
        OR (d.icd_version = 9  AND STARTS_WITH(d.icd_code,'414')),
        1, 0)
    ) AS CAD,
    MAX(
      IF(
        (d.icd_version = 10 AND STARTS_WITH(d.icd_code,'I27'))
        OR (d.icd_version = 9  AND STARTS_WITH(d.icd_code,'416')),
        1, 0)
    ) AS PH,
    MAX(
      IF(
        (d.icd_version = 10 AND STARTS_WITH(d.icd_code,'I48'))
        OR (d.icd_version = 9  AND (
              STARTS_WITH(d.icd_code,'42731') OR
              STARTS_WITH(d.icd_code,'42732')
            )),
        1, 0)
    ) AS AF
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
  JOIN coh c USING (subject_id, hadm_id)
  GROUP BY c.subject_id, c.hadm_id
),

/* ---------------- Fuse vitals: ED (agg) → ED triage → ICU; BMI: OMR → ICU ---------------- */
vitals_fused AS (
  SELECT
    d.subject_id,
    d.hadm_id,
    COALESCE(ea.ed_hr_mean,  et.ed_heartrate,  iv.icu_hr_mean)       AS heart_rate,
    COALESCE(ea.ed_sbp_mean, et.ed_sbp,        iv.icu_sbp_mean)      AS systolic_bp,
    COALESCE(ea.ed_spo2_mean, et.ed_spo2,      iv.icu_spo2_mean)     AS oxygen_saturation,
    COALESCE(ea.ed_temp_mean, et.ed_temperature_c, iv.icu_temp_mean) AS temperature,
    COALESCE(ob.omr_bmi, iv.icu_bmi)             AS bmi
  FROM demo d
  LEFT JOIN ed_triage  et USING (subject_id, hadm_id)
  LEFT JOIN ed_vs_agg  ea USING (subject_id, hadm_id)
  LEFT JOIN icu_vitals iv USING (subject_id, hadm_id)
  LEFT JOIN omr_bmi    ob USING (subject_id, hadm_id)
)

/* ---------------- Final feature table ---------------- */
SELECT
  d.subject_id,
  d.hadm_id,

  -- targets
  o.death_30_days,
  o.death_1_year,

  -- demographics
  d.age_admission,
  d.gender,

  -- vitals (fused)
  v.temperature,
  v.heart_rate,
  v.oxygen_saturation,
  v.systolic_bp,
  v.bmi,

  -- labs (COALESCE of derived + raw)
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

  -- comorbidities (Charlson components)
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
  c.moderate_severe_liver_disease,
  c.metastatic_cancer,

  -- dx flags
  e.HT,
  e.CAD,
  e.PH,
  e.AF

FROM demo d
LEFT JOIN outcomes     o USING (subject_id, hadm_id)
LEFT JOIN vitals_fused v USING (subject_id, hadm_id)
LEFT JOIN labs         l USING (subject_id, hadm_id)
LEFT JOIN charlson     c USING (hadm_id)
LEFT JOIN extras       e USING (subject_id, hadm_id);