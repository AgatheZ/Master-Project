--- Create a table containing the stay id of TBI patients 
-- DROP TABLE IF EXISTS mimiciv.TBI;
-- CREATE TABLE mimiciv.TBI AS 
-- (

-- SELECT icu.stay_id
-- FROM mimic_hosp.diagnoses_icd v
-- JOIN mimic_hosp.d_icd_diagnoses d 
-- ON v.icd_code = d.icd_code
-- JOIN mimic_icu.icustays icu 
-- ON v.subject_id = icu.subject_id
-- WHERE (icu.los >= 2) AND ((d.icd_code LIKE '850%') OR (d.icd_code LIKE '851%') OR (d.icd_code LIKE '852%') OR (d.icd_code LIKE '853%') OR (d.icd_code LIKE '854%') OR (d.icd_code LIKE '959.01%'))
-- ); -- Only take patients with records > 48h

--- Create a table containing the stay id of TBI patients - AUGMENTED VERSION
DROP TABLE IF EXISTS mimiciv.TBI;
CREATE TABLE mimiciv.TBI AS 
(

SELECT icu.stay_id
FROM mimic_hosp.diagnoses_icd v
JOIN mimic_hosp.d_icd_diagnoses d 
ON v.icd_code = d.icd_code
JOIN mimic_icu.icustays icu 
ON v.subject_id = icu.subject_id
WHERE (icu.los >= 2) AND ((d.icd_code LIKE '850%') OR (d.icd_code LIKE '851%') OR (d.icd_code LIKE '852%') OR (d.icd_code LIKE '853%') OR (d.icd_code LIKE '854%') OR (d.icd_code LIKE '959.01%')
 OR (d.icd_code LIKE '80%') OR (d.icd_code LIKE '81%') OR (d.icd_code LIKE '82%') OR (d.icd_code LIKE '83%') OR (d.icd_code LIKE '84%') 
 OR (d.icd_code LIKE '86%') OR (d.icd_code LIKE '87%') OR (d.icd_code LIKE '88%') OR (d.icd_code LIKE '89%') OR (d.icd_code LIKE '90%')
 OR (d.icd_code LIKE '91%')  OR (d.icd_code LIKE '92%') OR (d.icd_code LIKE '93%') OR (d.icd_code LIKE '94%') OR (d.icd_code LIKE '95%') OR (d.icd_code LIKE '96%') )
); -- Only take patients with records > 48h

---Creation of the vitals lookup table 
DROP TABLE IF EXISTS mimiciv.lookup CASCADE;
CREATE TABLE mimiciv.lookup (vital_name varchar(50), aggregation int);
COPY mimiciv.lookup(vital_name, aggregation) FROM 'lookup_vitals_aggregation_aug.csv' delimiter ',' CSV HEADER;


--- Filter the vitals we want to extract 
DROP TABLE IF EXISTS mimiciv.cohort_vitals;
CREATE TABLE mimiciv.cohort_vitals
AS

SELECT
ce.stay_id, fl.abbreviation as vital_name, ce.charttime, ce.value as vital_reading, d.long_title as diagnosis

FROM mimic_icu.chartevents ce 
JOIN mimic_icu.d_items fl 
ON fl.itemid = ce.itemid
JOIN mimiciv.lookup lk 
ON lk.vital_name = fl.abbreviation
JOIN mimic_hosp.diagnoses_icd v
ON v.hadm_id = ce.hadm_id
JOIN mimic_hosp.d_icd_diagnoses d 
ON v.icd_code = d.icd_code
JOIN mimiciv.TBI tbi ON
ce.stay_id = tbi.stay_id

WHERE ( lk.aggregation = '1') AND ((fl.param_type like '%Numeric%')) AND (((fl.category != 'Alarms') AND (fl.category != 'General')));

--- Data aggregation - HOURLY ---------------------------------------------------------------------------------------
DROP TABLE IF EXISTS  mimiciv.aggregated_vitals_hourly CASCADE;
CREATE TABLE  mimiciv.aggregated_vitals_hourly
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
FROM
mimiciv.cohort_vitals vit
LEFT JOIN  mimic_icu.icustays icu
ON vit.stay_id = icu.stay_id
JOIN mimiciv.lookup lk
ON vit.vital_name = lk.vital_name
WHERE lk.aggregation = '1' AND icu.los >= 1.04


) , 

aggregated 
AS

(
SELECT
stay_id
,icu_intime
,EXTRACT(DAY FROM diff_chart_intime)*24    + EXTRACT(HOUR FROM diff_chart_intime) + case when  EXTRACT(MINUTE FROM diff_chart_intime) >=1 then 1 else 0 end as hour_from_intime -- number of hours from icu admitted time
,vital_name AS feature_name
, avg(vital_reading) AS feature_mean_value
FROM icu_vital_data
GROUP BY stay_id, icu_intime, hour_from_intime, feature_name
)

SELECT stay_id, icu_intime,  feature_name, feature_mean_value, hour_from_intime
FROM aggregated
GROUP BY stay_id, icu_intime, hour_from_intime, feature_name, feature_mean_value
ORDER BY stay_id, hour_from_intime;


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
JOIN mimiciv.TBI ON
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

---Intersect the stays of all the tables to have the final cohort 
DROP TABLE IF EXISTS mimiciv.final_stays;
CREATE TABLE mimiciv.final_stays
AS



SELECT stay_id
FROM mimiciv.aggregated_vitals_hourly
INTERSECT
SELECT stay_id
FROM mimiciv.demographics; 


---Get final vital tables 
--hourly
DROP TABLE IF EXISTS mimiciv.final_h;
CREATE TABLE mimiciv.final_h
AS

SELECT t.* from mimiciv.aggregated_vitals_hourly t
JOIN mimiciv.final_stays s 
ON s.stay_id = t.stay_id;

---demographics
DROP TABLE IF EXISTS mimiciv.final_dem;
CREATE TABLE mimiciv.final_dem
AS

SELECT t.* from mimiciv.demographics t
JOIN mimiciv.final_stays s 
ON s.stay_id = t.stay_id;

---print the final tables
SELECT * FROM mimiciv.final_dem;
SELECT * FROM mimiciv.final_h;
