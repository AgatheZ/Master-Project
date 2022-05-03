DROP MATERIALIZED VIEW IF EXISTS patient_info CASCADE;
CREATE MATERIALIZED VIEW patient_info as

SELECT ie.subject_id, ie.hadm_id, ie.icustay_id

-- patient level factors
, pat.gender

-- hospital level factors
, ROUND((CAST(EXTRACT(epoch FROM adm.admittime - pat.dob)/(60*60*24*365.242) AS numeric)), 0) AS age
, adm.admission_type

-- icu level factors
, ie.intime, ie.outtime
, ROUND((CAST(EXTRACT(epoch FROM ie.outtime - ie.intime)/(60*60*24) AS numeric)), 4) AS length_stayICU

, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN True
    ELSE False END AS first_icu_stay

-- diagnosis
, diag.short_title AS diagnosis

FROM mimiciiid.icustays ie

INNER JOIN mimiciiid.admissions adm
    ON ie.hadm_id = adm.hadm_id
INNER JOIN mimiciiid.patients pat
    ON ie.subject_id = pat.subject_id
INNER JOIN mimiciiid.diagnoses_icd jt
    ON ie.hadm_id = jt.hadm_id
INNER JOIN mimiciiid.d_icd_diagnoses diag
    ON jt.icd9_code = diag.icd9_code

--- only select the most important diagnosis 
WHERE jt.seq_num = 1

ORDER BY ie.subject_id, adm.admittime, ie.intime


;