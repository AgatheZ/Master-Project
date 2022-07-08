---Get demographic table 
DROP TABLE IF EXISTS mimiciv.demographics;
CREATE TABLE mimiciv.demographics AS


WITH transformed 
AS
(

SELECT ce.stay_id, fl.abbreviation as vital_name,

CASE  WHEN ce.value IN ('None', 'No response', 'No Response', 'No Response-ETT') THEN 1
WHEN ce.value IN ('To Pain', 'Abnormal extension', 'Incomprehensible sounds') THEN 2 
WHEN ce.value IN ('To Speech', 'Abnormal Flexion', 'Inappropriate Words') THEN 3 
WHEN ce.value IN ('Spontaneously', 'Flex-withdraws', 'Confused') THEN 4 
WHEN ce.value IN ('Localizes Pain', 'Oriented') THEN 5 
WHEN ce.value IN ('Obey Commands') THEN 6 END AS reading,

ce.charttime

FROM mimic_icu.chartevents ce 
JOIN mimic_icu.d_items fl 
ON fl.itemid = ce.itemid
RIGHT JOIN mimiciv.TBI ON
ce.stay_id = tbi.stay_id
WHERE fl.itemid IN (220739, 223900, 223901) 
),

averaged AS 
(
SELECT stay_id, vital_name, avg(reading) as reading
FROM transformed 
GROUP BY stay_id, vital_name

),

rest AS
(
SELECT DISTINCT v.stay_id, p.gender, p.anchor_age AS age, icu.los as LOS,  ROUND(hw.weight_first / POWER(hw.height_first / 100, 2), 3) AS BMI, case when p.dod is not null and p.dod <= icu.outtime then 1 else 0 end as death
FROM mimiciv.TBI v 
JOIN mimic_icu.icustays icu ON v.stay_id = icu.stay_id
JOIN mimic_core.patients p ON p.subject_id = icu.subject_id
LEFT JOIN mimiciv.heightweight hw ON hw.stay_id = icu.stay_id
ORDER BY v.stay_id 

)

SELECT rest.stay_id, rest.gender, rest.age, rest.los, rest.bmi, rest.death, ROUND(sum(reading),0) AS GCS 
FROM rest, averaged  
WHERE rest.stay_id = averaged.stay_id
GROUP BY rest.stay_id, rest.gender, rest.age, rest.los, rest.bmi, rest.death
;
