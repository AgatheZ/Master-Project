-- aggregation per minute 
DROP TABLE IF EXISTS  mimiciiid.pivoted CASCADE;
CREATE TABLE  mimiciiid.pivoted
AS

SELECT  icustay_id, icu_intime, minute_from_intime, 
feature_mean_value FILTER (WHERE feature_name = 'Heart Rate' ) AS 'Heart Rate',
feature_mean_value FILTER (WHERE feature_name = 'Oxygen Saturation' ) AS 'Oxygen Saturation',
feature_mean_value FILTER (WHERE feature_name = 'Mean Arterial Pressure' ) AS 'Mean Arterial Pressure ',
feature_mean_value FILTER (WHERE feature_name = 'Intracranial Pressure') AS 'Intracranial Pressure'

FROM mimiciiid.aggregated_vitals
GROUP BY icustay_id, icu_intime, minute_from_intime
;