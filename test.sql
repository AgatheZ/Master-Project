--- HOURLY AGGREGATION -----------------------------------------------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS  mimiciv.aggregated_vitals_h CASCADE;
CREATE TABLE  mimiciv.aggregated_vitals_h
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
JOIN lookup_vitals_aggregation lp
ON vit.vital_name = lp.vital_name


WHERE icu.los >= 2 AND lp.aggregation = 1) , -- Only take patients with records > 48h


aggregated 
AS

(
SELECT
stay_id
,icu_intime

CASE WHEN 
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

--------- 24 hours AGGREGATION --------------------------------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS  mimiciv.aggregated_vitals_24h CASCADE;
CREATE TABLE  mimiciv.aggregated_vitals_24h
AS

WITH icu_vital_data
AS

SELECT
vit.stay_id, DATE_TRUNC('day', icu.intime) as icu_intime 
,vit.charttime - DATE_TRUNC('day', icu.intime) as diff_chart_intime -- difference between charttime and icu admitted time
,vit.vital_name
, CAST(vit.vital_reading AS NUMERIC)
, vit.charttime AS charttime
, vit.diagnosis
FROM
mimiciv.cohort_vitals vit
LEFT JOIN  mimic_icu.icustays icu
ON vit.stay_id = icu.stay_id
JOIN lookup_vitals_aggregation lp
ON vit.vital_name = lp.vital_name


WHERE icu.los >= 2 AND lp.aggregation = 24) , -- Only take patients with records > 48h


aggregated 
AS

(
SELECT
stay_id
,icu_intime

CASE WHEN 
,EXTRACT(DAY FROM diff_chart_intime) +  case when  EXTRACT(DAY FROM diff_chart_intime) >=1 then 1 else 0 end as day_from_intime -- number of hours from icu admitted time
,vital_name AS feature_name
,avg(vital_reading) AS feature_mean_value
, diagnosis
FROM icu_vital_data
GROUP BY stay_id, icu_intime, day_from_intime, feature_name, diagnosis
)

SELECT stay_id, icu_intime,  feature_name, feature_mean_value, diagnosis,
    CASE WHEN hour_from_intime < 0 THEN 0
    ELSE hour_from_intime END
    FROM aggregated
    GROUP BY stay_id, icu_intime, hour_from_intime, feature_name, feature_mean_value, diagnosis
    ORDER BY stay_id, day_from_intime;

--------- 48 hours AGGREGATION --------------------------------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS  mimiciv.aggregated_vitals_48h CASCADE;
CREATE TABLE  mimiciv.aggregated_vitals_48h
AS

WITH icu_vital_data
AS

SELECT
vit.stay_id, DATE_TRUNC('day', icu.intime) as icu_intime 
,vit.charttime - DATE_TRUNC('day', icu.intime) as diff_chart_intime -- difference between charttime and icu admitted time
,vit.vital_name
, CAST(vit.vital_reading AS NUMERIC)
, vit.charttime AS charttime
, vit.diagnosis
FROM
mimiciv.cohort_vitals vit
LEFT JOIN  mimic_icu.icustays icu
ON vit.stay_id = icu.stay_id
JOIN lookup_vitals_aggregation lp
ON vit.vital_name = lp.vital_name


WHERE icu.los >= 2 AND lp.aggregation = 48) , -- Only take patients with records > 48h


aggregated 
AS

(
SELECT
stay_id
,icu_intime

CASE WHEN 
,EXTRACT(DAY FROM diff_chart_intime) +  case when  EXTRACT(DAY FROM diff_chart_intime) >=1 then 1 else 0 end as day_from_intime -- number of hours from icu admitted time
,vital_name AS feature_name
,avg(vital_reading) AS feature_mean_value
, diagnosis
FROM icu_vital_data
GROUP BY stay_id, icu_intime, day_from_intime, feature_name, diagnosis
)

SELECT stay_id, icu_intime,  feature_name, feature_mean_value, diagnosis,
    CASE WHEN hour_from_intime < 0 THEN 0
    ELSE hour_from_intime END
    FROM aggregated
    GROUP BY stay_id, icu_intime, hour_from_intime, feature_name, feature_mean_value, diagnosis
    ORDER BY stay_id, day_from_intime;