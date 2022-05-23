--- Filter the vitals using the lookup.csv we created filter the patients to only get TBI patients 
DROP TABLE IF EXISTS mimiciii.lookup CASCADE;
CREATE TABLE mimiciii.lookup (final varchar(50), original varchar(50), item_code integer, units varchar(10));
COPY mimiciii.lookup(final, original, item_code, units) FROM 'lookup_table.csv' delimiter ',' CSV HEADER;


DROP TABLE IF EXISTS mimiciii.cohort_vitals;
CREATE TABLE mimiciii.cohort_vitals
AS

SELECT
ce.icustay_id,ce.charttime ,fl.final as vital_name, ROUND(CAST(ce.value as numeric), 2)  as vital_reading, adm.diagnosis as diagnosis

FROM mimiciii.chartevents ce 
JOIN mimiciii.lookup fl 
ON fl.item_code = ce.itemid
JOIN mimiciii.admissions adm
ON adm.hadm_id = ce.hadm_id 

WHERE diagnosis LIKE '%HEAD INJURY%' OR diagnosis LIKE '%HEAD TRAUMA%';

--- Creating a table that contains vitals range using the vital_range.csv we created 
DROP TABLE IF EXISTS mimiciii.vital_range CASCADE;
CREATE TABLE mimiciii.vital_range (vital varchar(50), low integer, high integer);
COPY mimiciii.vital_range(vital, low, high) FROM 'vitals_range.csv' delimiter ',' CSV HEADER;

--- Remove the outliers using the vital_range table we just created
DROP TABLE IF EXISTS  mimiciii.vitals_w_outliers CASCADE;
CREATE TABLE  mimiciii.vitals_w_outliers
AS

SELECT
vs.icustay_id, vs.charttime, vs.vital_name, vs.diagnosis as diagnosis,
CASE
WHEN CAST(vs.vital_reading AS integer) < low or CAST(vs.vital_reading AS integer)> high then Null
ELSE CAST(vs.vital_reading AS integer)
END AS outlier_handled_vital_reading 

FROM mimiciii.cohort_vitals vs
LEFT JOIN mimiciii.vital_range vr
ON lower(vs.vital_name) = lower(vr.vital);

--- Data aggregation (per minute as advised by Dr. Libert)
DROP TABLE IF EXISTS  mimiciii.aggregated_vitals CASCADE;
CREATE TABLE  mimiciii.aggregated_vitals
AS

WITH icu_vital_data
AS

(
SELECT
vit.icustay_id, DATE_TRUNC('minute', icu.intime) as icu_intime 
,vit.charttime - DATE_TRUNC('minute', icu.intime) as diff_chart_intime -- difference between charttime and icu admitted time
,vit.vital_name
,vit.outlier_handled_vital_reading
, vit.charttime AS charttime
, vit.diagnosis
FROM
mimiciii.vitals_w_outliers vit
LEFT JOIN  mimiciii.icustays icu
ON vit.icustay_id = icu.ICUSTAY_ID
WHERE icu.los >= 1), -- Only take patients with records > 24h

aggregated 
AS

(
SELECT
icustay_id
,icu_intime
,EXTRACT(DAY FROM diff_chart_intime) * 24 + EXTRACT(HOUR FROM diff_chart_intime)  +  case when  EXTRACT(MINUTE FROM diff_chart_intime) >=1 then 1 else 0 end as hour_from_intime -- number of hours from icu admitted time
,vital_name AS feature_name
,avg(outlier_handled_vital_reading) AS feature_mean_value
, diagnosis
FROM icu_vital_data
GROUP BY icustay_id, icu_intime, hour_from_intime, feature_name, diagnosis
)

SELECT icustay_id, icu_intime,  feature_name, feature_mean_value, diagnosis,
CASE WHEN hour_from_intime < 0 THEN 0
ELSE hour_from_intime END
FROM aggregated
GROUP BY icustay_id, icu_intime, hour_from_intime, feature_name, feature_mean_value, diagnosis
ORDER BY icustay_id, hour_from_intime;


--- Pivot the data for better readability (NEEDS TO BE CHANGED IF WE CHANGE THE VARIABLES)
DROP TABLE IF EXISTS  mimiciii.preprocessed CASCADE;
CREATE TABLE  mimiciii.preprocessed
AS
(
WITH transition 
AS
(
SELECT icustay_id,   icu_intime,  hour_from_intime , diagnosis, 
    round(CASE WHEN feature_name = 'Heart Rate' THEN feature_mean_value END, 2) AS "Heart Rate",
    round(CASE WHEN feature_name = 'Oxygen Saturation' THEN feature_mean_value END, 2) AS "Oxygen Saturation",
    round(CASE WHEN feature_name = 'Mean Arterial Pressure ' THEN feature_mean_value END, 2) AS "MAP",
    round(CASE WHEN feature_name = 'Intracranial Pressure' THEN feature_mean_value END, 2) AS "Intracranial Pressure",
    round(CASE WHEN feature_name = 'CPP' THEN feature_mean_value END, 2) AS "CPP"


FROM mimiciii.aggregated_vitals
ORDER BY icustay_id, hour_from_intime
)

SELECT  icustay_id,     icu_intime     ,  hour_from_intime  , diagnosis, sum("Heart Rate") AS Heart_rate , sum("Oxygen Saturation") AS Oxygen_saturation, sum("MAP") AS MAP , sum("Intracranial Pressure") AS ICP, sum("CPP") AS CPP
FROM transition
GROUP BY  icustay_id,     icu_intime     ,  hour_from_intime, diagnosis
ORDER BY icustay_id, hour_from_intime
); 

SELECT * FROM mimiciii.preprocessed order by icustay_id, hour_from_intime;
-- Don't forget to save it again as a csv if changes are made to the pipeline 



