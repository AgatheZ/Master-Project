--- Create a table containing the stay id of TBI patients 
DROP TABLE IF EXISTS mimiciv.TBI;
CREATE TABLE mimiciv.TBI AS 
(

SELECT icu.stay_id
FROM mimic_hosp.diagnoses_icd v
JOIN mimic_hosp.d_icd_diagnoses d 
ON v.icd_code = d.icd_code
JOIN mimic_icu.icustays icu 
ON v.subject_id = icu.subject_id
WHERE  (d.icd_code LIKE '850%') OR (d.icd_code LIKE '851%') OR (d.icd_code LIKE '852%') OR (d.icd_code LIKE '853%') OR (d.icd_code LIKE '854%')
);

--- Filter the vitals we want to extract 
DROP TABLE IF EXISTS mimiciv.cohort_vitals;
CREATE TABLE mimiciv.cohort_vitals
AS

SELECT
ce.stay_id, fl.abbreviation as vital_name, ce.charttime, ce.value as vital_reading, d.long_title as diagnosis

FROM mimic_icu.chartevents ce 
JOIN mimic_icu.d_items fl 
ON fl.itemid = ce.itemid
JOIN mimic_hosp.diagnoses_icd v
ON v.hadm_id = ce.hadm_id
JOIN mimic_hosp.d_icd_diagnoses d 
ON v.icd_code = d.icd_code
JOIN mimiciv.TBI tbi ON
ce.stay_id = tbi.stay_id

WHERE ((fl.param_type = 'Numeric'))AND (((fl.category != 'Alarms') AND (fl.category != 'General')));

--- Outlier removal that we dont do for now 
/* Creating a table that contains vitals range using the vital_range.csv we created 
---DROP TABLE IF EXISTS mimiciv.vital_range CASCADE;
---CREATE TABLE mimiciv.vital_range (vital varchar(50), low integer, high integer);
---COPY mimiciv.vital_range(vital, low, high) FROM 'vitals_range.csv' delimiter ',' CSV HEADER;

--- Remove the outliers using the vital_range table we just created
---DROP TABLE IF EXISTS  mimiciv.vitals_w_outliers CASCADE;
CREATE TABLE  mimiciv.vitals_w_outliers
AS

SELECT
vs.icustay_id, vs.charttime, vs.vital_name, vs.diagnosis as diagnosis,
CASE
WHEN CAST(vs.vital_reading AS integer) < low or CAST(vs.vital_reading AS integer)> high then Null
ELSE CAST(vs.vital_reading AS integer)
END AS outlier_handled_vital_reading 

FROM mimiciv.cohort_vitals vs
LEFT JOIN mimiciv.vital_range vr
ON lower(vs.vital_name) = lower(vr.vital); */

--- Data aggregation (per hour as advised by Dr. Libert)

DROP TABLE IF EXISTS  mimiciv.aggregated_vitals CASCADE;
CREATE TABLE  mimiciv.aggregated_vitals
AS

WITH icu_vital_data
AS

(
SELECT
vit.stay_id, DATE_TRUNC('hour', icu.intime) as icu_intime 
,vit.charttime - DATE_TRUNC('hour', icu.intime) as diff_chart_intime -- difference between charttime and icu admitted time
,vit.vital_name
, CAST(vit.vital_reading AS NUMERIC)
, vit.charttime AS charttime
, vit.diagnosis
FROM
mimiciv.cohort_vitals vit
LEFT JOIN  mimic_icu.icustays icu
ON vit.stay_id = icu.stay_id
WHERE icu.los >= 2) , -- Only take patients with records > 48h


aggregated 
AS

(
SELECT
stay_id
,icu_intime
,EXTRACT(DAY FROM diff_chart_intime) * 24 + EXTRACT(HOUR FROM diff_chart_intime)  +  case when  EXTRACT(MINUTE FROM diff_chart_intime) >=1 then 1 else 0 end as hour_from_intime -- number of hours from icu admitted time
,vital_name AS feature_name
,avg(vital_reading) AS feature_mean_value
, diagnosis
FROM icu_vital_data
GROUP BY stay_id, icu_intime, hour_from_intime, feature_name, diagnosis
)

SELECT stay_id, icu_intime,  feature_name, feature_mean_value, diagnosis,
CASE WHEN hour_from_intime < 0 THEN 0
ELSE hour_from_intime END
FROM aggregated
GROUP BY stay_id, icu_intime, hour_from_intime, feature_name, feature_mean_value, diagnosis
ORDER BY stay_id, hour_from_intime;


---Get demographic table 
DROP TABLE IF EXISTS mimiciv.demographics;
CREATE TABLE mimiciv.demographics AS
(

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
JOIN mimiciv.TBI ON
ce.stay_id = tbi.stay_id
WHERE fl.itemid IN (220739, 223900, 223901) 
),

averaged AS 
(
SELECT stay_id, vital_name, avg(reading) as reading
FROM transformed 
GROUP BY stay_id, vital_name
)

SELECT DISTINCT v.stay_id, p.gender, p.anchor_age AS age, ROUND(hw.weight_first / POWER(hw.height_first / 100, 2), 3) AS BMI, case when p.dod is not null and p.dod <= icu.outtime then 1 else 0 end as death, ROUND(sum(reading),0) AS GCS
FROM averaged a 
JOIN mimiciv.aggregated_vitals v 
ON v.stay_id = a.stay_id
JOIN mimic_icu.icustays icu ON v.stay_id = icu.stay_id
JOIN mimic_core.patients p ON p.subject_id = icu.subject_id
LEFT JOIN mimiciv.heightweight hw ON hw.stay_id = icu.stay_id
ORDER BY v.stay_id 
);


---Get vitals table 
DROP TABLE IF EXISTS mimiciv.preprocessed_vitals;
CREATE TABLE mimiciv.preprocessed_vitals AS
(
SELECT stay_id, icu_intime, feature_name, round(feature_mean_value, 2), hour_from_intime FROM mimiciv.aggregated_vitals
);