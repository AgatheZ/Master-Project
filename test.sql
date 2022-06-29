SELECT icu.stay_id
FROM mimic_hosp.diagnoses_icd v
JOIN mimic_hosp.d_icd_diagnoses d 
ON v.icd_code = d.icd_code
JOIN mimic_icu.icustays icu 
ON v.subject_id = icu.subject_id
WHERE d.icd_code LIKE '851%' AND (icu.los >= 2)