DROP TABLE IF EXISTS  mimiciiid.aggregated_vitals_hour CASCADE;
CREATE TABLE  mimiciiid.aggregated_vitals_hour
AS

WITH icu_vital_data
AS

(
SELECT
vit.icustay_id, DATE_TRUNC('hour', icu.intime) as icu_intime 
,vit.charttime - DATE_TRUNC('hour', icu.intime) as diff_chart_intime -- difference between charttime and icu admitted time
,vit.vital_name
,vit.outlier_handled_vital_reading
, vit.charttime AS charttime
FROM
mimiciiid.vitals_w_outliers vit
LEFT JOIN  mimiciiid.icustays icu
ON vit.icustay_id = icu.ICUSTAY_ID
),

aggregated 
AS

(
SELECT
icustay_id
,icu_intime
,EXTRACT(DAY FROM diff_chart_intime) * 24 + EXTRACT(HOUR FROM diff_chart_intime)  +  case when  EXTRACT(MINUTE FROM diff_chart_intime) >=1 then 1 else 0 end as hour_from_intime -- number of hours from icu admitted time
,vital_name AS feature_name
,avg(outlier_handled_vital_reading) AS feature_mean_value
FROM icu_vital_data
GROUP BY icustay_id, icu_intime, hour_from_intime, feature_name
)

SELECT icustay_id, icu_intime,  feature_name, feature_mean_value,
CASE WHEN hour_from_intime < 0 THEN 0
ELSE hour_from_intime END
FROM aggregated
GROUP BY icustay_id, icu_intime, hour_from_intime, feature_name, feature_mean_value
ORDER BY icustay_id, hour_from_intime;


--- Pivot the data for better readability (NEEDS TO BE CHANGED IF WE CHANGE THE VARIABLES)
DROP TABLE IF EXISTS  mimiciiid.preprocessed_hour CASCADE;
CREATE TABLE  mimiciiid.preprocessed_hour
AS
(
WITH transition 
AS
(
SELECT icustay_id,   icu_intime,  hour_from_intime ,
    round(CASE WHEN feature_name = 'Heart Rate' THEN feature_mean_value END, 2) AS "Heart Rate",
    round(CASE WHEN feature_name = 'Oxygen Saturation' THEN feature_mean_value END, 2) AS "Oxygen Saturation",
    round(CASE WHEN feature_name = 'Mean Arterial Pressure ' THEN feature_mean_value END, 2) AS "MAP",
    round(CASE WHEN feature_name = 'Intracranial Pressure' THEN feature_mean_value END, 2) AS "Intracranial Pressure"

FROM mimiciiid.aggregated_vitals_hour
ORDER BY icustay_id, hour_from_intime
)

SELECT  icustay_id,     icu_intime     ,  hour_from_intime  , sum("Heart Rate") AS Heart_rate , sum("Oxygen Saturation") AS Oxygen_saturation, sum("MAP") AS MAP , sum("Intracranial Pressure") AS ICP
FROM transition
GROUP BY  icustay_id,     icu_intime     ,  hour_from_intime
ORDER BY icustay_id, hour_from_intime
); 

SELECT * FROM mimiciiid.preprocessed_hour order by icustay_id, hour_from_intime;




