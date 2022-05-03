SELECT COUNT(vrai.hadm_id), fake.short_title from mimiciiid.diagnoses_icd as vrai 
INNER JOIN mimiciiid.d_icd_diagnoses as fake
ON vrai.icd9_code = fake.icd9_code
WHERE fake.short_title LIKE '%hemo%'
GROUP BY fake.short_title