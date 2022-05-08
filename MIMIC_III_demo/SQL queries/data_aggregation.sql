-- aggregation per minute 
DROP TABLE IF EXISTS  mimiciiid.aggregated_vitals CASCADE;
CREATE TABLE  mimiciiid.aggregated_vitals
AS

WITH icu_vital_data
AS

(
SELECT
vit.icustay_id
,DATE_TRUNC('minute', icu.intime) as icu_intime -- round it to the nearest hour
,vit.charttime - DATE_TRUNC('minute', icu.intime) as diff_chart_intime -- difference between charttime and icu admitted time
,vit.vital_name
,vit.outlier_handled_vital_reading
, vit.charttime as charttime

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



