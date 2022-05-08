--outlier handling for mimic-iii data
DROP TABLE IF EXISTS  mimiciiid.vitals_w_outliers CASCADE;
CREATE TABLE  mimiciiid.vitals_w_outliers
AS
SELECT
vs.icustay_id
, vs.charttime
, vs.vital_name --vital name
, 

CASE
WHEN CAST(vs.vital_reading AS integer) < low or CAST(vs.vital_reading AS integer)> high then Null
ELSE CAST(vs.vital_reading AS integer)
END AS outlier_handled_vital_reading --outlier corrected vital reading

FROM mimiciiid.cohort_vitals vs
LEFT JOIN mimiciiid.vital_range vr
ON lower(vs.vital_name) = lower(vr.vital);