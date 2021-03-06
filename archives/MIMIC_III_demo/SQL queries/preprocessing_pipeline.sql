--- Filter the vitals using the lookup.csv we created, filter the patients to only get TBI patients 
DROP TABLE IF EXISTS mimiciiid.lookup CASCADE;
CREATE TABLE mimiciiid.lookup (final varchar(50), original varchar(50), item_code integer, units varchar(10));
COPY mimiciiid.lookup(final, original, item_code, units) FROM 'lookup_table.csv' delimiter ',' CSV HEADER;


DROP TABLE IF EXISTS mimiciiid.cohort_vitals;
CREATE TABLE mimiciiid.cohort_vitals
AS

SELECT
ce.icustay_id,ce.charttime ,fl.final as vital_name, ROUND(CAST(ce.value as numeric), 2)  as vital_reading 

FROM mimiciiid.chartevents ce 
JOIN mimiciiid.lookup fl 
ON fl.item_code = ce.itemid ;

--- Creating a table that contains vitals range using the vital_range.csv we created 
DROP TABLE IF EXISTS mimiciiid.vital_range CASCADE;
CREATE TABLE mimiciiid.vital_range (vital varchar(50), low integer, high integer);
COPY mimiciiid.vital_range(vital, low, high) FROM 'vitals_range.csv' delimiter ',' CSV HEADER;

--- Remove the outliers using the vital_range table we just created
DROP TABLE IF EXISTS  mimiciiid.vitals_w_outliers CASCADE;
CREATE TABLE  mimiciiid.vitals_w_outliers
AS

SELECT
vs.icustay_id, vs.charttime, vs.vital_name, 
CASE
WHEN CAST(vs.vital_reading AS integer) < low or CAST(vs.vital_reading AS integer)> high then Null
ELSE CAST(vs.vital_reading AS integer)
END AS outlier_handled_vital_reading 

FROM mimiciiid.cohort_vitals vs
LEFT JOIN mimiciiid.vital_range vr
ON lower(vs.vital_name) = lower(vr.vital);

--- Data aggregation (per minute as advised by Dr. Libert)
DROP TABLE IF EXISTS  mimiciiid.aggregated_vitals CASCADE;
CREATE TABLE  mimiciiid.aggregated_vitals
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
FROM
mimiciiid.vitals_w_outliers vit
LEFT JOIN  mimiciiid.icustays icu
ON vit.icustay_id = icu.ICUSTAY_ID
WHERE icu.los >= 1), -- Only take patients with records > 24h

aggregated 
AS

(
SELECT
icustay_id
,icu_intime
,EXTRACT(DAY FROM diff_chart_intime) * 1440 + EXTRACT(HOUR FROM diff_chart_intime) * 60 + EXTRACT(MINUTE FROM diff_chart_intime) + case when  EXTRACT(SECOND from diff_chart_intime) >=1 then 1 else 0 end as minute_from_intime -- number of minutes from icu admitted time
,vital_name AS feature_name
,avg(outlier_handled_vital_reading) AS feature_mean_value
FROM icu_vital_data
GROUP BY icustay_id, icu_intime, minute_from_intime, feature_name
)

SELECT icustay_id, icu_intime,  feature_name, feature_mean_value,
CASE WHEN minute_from_intime < 0 THEN 0
ELSE minute_from_intime END
FROM aggregated
GROUP BY icustay_id, icu_intime, minute_from_intime, feature_name, feature_mean_value
ORDER BY icustay_id, minute_from_intime;


--- Pivot the data for better readability (NEEDS TO BE CHANGED IF WE CHANGE THE VARIABLES)
DROP TABLE IF EXISTS  mimiciiid.preprocessed CASCADE;
CREATE TABLE  mimiciiid.preprocessed
AS
(
WITH transition 
AS
(
SELECT icustay_id,   icu_intime,  minute_from_intime ,
    round(CASE WHEN feature_name = 'Heart Rate' THEN feature_mean_value END, 2) AS "Heart Rate",
    round(CASE WHEN feature_name = 'Oxygen Saturation' THEN feature_mean_value END, 2) AS "Oxygen Saturation",
    round(CASE WHEN feature_name = 'Mean Arterial Pressure ' THEN feature_mean_value END, 2) AS "MAP",
    round(CASE WHEN feature_name = 'Intracranial Pressure' THEN feature_mean_value END, 2) AS "Intracranial Pressure",
    round(CASE WHEN feature_name = 'CPP' THEN feature_mean_value END, 2) AS "CPP"


FROM mimiciiid.aggregated_vitals
ORDER BY icustay_id, minute_from_intime
)

SELECT  icustay_id,     icu_intime     ,  minute_from_intime  , sum("Heart Rate") AS Heart_rate , sum("Oxygen Saturation") AS Oxygen_saturation, sum("MAP") AS MAP , sum("Intracranial Pressure") AS ICP, sum("CPP") AS CPP
FROM transition
GROUP BY  icustay_id,     icu_intime     ,  minute_from_intime
ORDER BY icustay_id, minute_from_intime
); 

SELECT * FROM mimiciiid.preprocessed order by icustay_id, minute_from_intime;
-- Don't forget to save it again as a csv if changes are made to the pipeline 



