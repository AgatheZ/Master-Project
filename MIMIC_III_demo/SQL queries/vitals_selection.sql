CREATE TABLE mimiciiid.cohort_vitals
AS

SELECT
-- icu stay id
ce.icustay_id -- icu stay id
,ce.charttime -- date/time when reading was taken
,fl.final as vital_name -- vital name
,ce.value as vital_reading -- vital reading

FROM mimiciiid.chartevents ce 

JOIN mimiciiid.lookup fl 
ON fl.item_code = ce.itemid ;
