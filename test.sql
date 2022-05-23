SELECT DISTINCT v.stay_id, p.gender, p.anchor_age AS age, ROUND(hw.weight_first / POWER(hw.height_first / 100, 2), 3) AS BMI
FROM mimiciv.aggregated_vitals v
JOIN mimic_icu.icustays icu ON v.stay_id = icu.stay_id
JOIN mimic_core.patients p ON p.subject_id = icu.subject_id
JOIN mimiciv.heightweight hw ON hw.stay_id = icu.stay_id

ORDER BY stay_id ;

