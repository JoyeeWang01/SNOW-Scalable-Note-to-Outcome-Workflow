-- ================================================================================
-- STEP 2: Add Discharge Notes to Cohort
-- ================================================================================
--
-- Purpose: Join discharge summaries from MIMIC-IV notes to the cohort
--
-- Input:
--   - Table from Step 1 (cohort of HFpEF patients)
--   - physionet-data.mimiciv_note.discharge (discharge summaries)
--
-- Output: Cohort table with discharge_text column added
--   - Concatenates multiple discharge notes per admission (if any)
--   - Separator: '\n\n-----\n\n'
--   - Ordered by charttime DESC (most recent first)
--
-- IMPORTANT: Update both table names below:
--   1. Output table name (line 22)
--   2. Input cohort table from Step 1 (line 33)
-- ================================================================================

-- TODO: Replace with your project and dataset name
CREATE OR REPLACE TABLE `your_project.your_dataset.hfpef_with_notes` AS
WITH discharge_concat AS (
  SELECT
    hadm_id,
    STRING_AGG(text, '\n\n-----\n\n' ORDER BY charttime DESC NULLS LAST, note_id DESC) AS discharge_text
  FROM `physionet-data.mimiciv_note.discharge`
  GROUP BY hadm_id
)
SELECT
  h.*,
  dn.discharge_text
-- TODO: Replace with your cohort table from Step 1
FROM `your_project.your_dataset.hfpef_cohort` AS h
INNER JOIN discharge_concat AS dn
  ON CAST(h.hadm_id AS INT64) = dn.hadm_id;