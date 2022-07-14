-- Script to extract all the available values for ABPm, ABPs, ABPd

SELECT
ce.stay_id, fl.abbreviation as vital_name, ce.value as vital_reading, 
EXTRACT(DAY FROM ce.charttime - DATE_TRUNC('hour', icu.intime))    + case when  EXTRACT(HOUR FROM ce.charttime - DATE_TRUNC('hour', icu.intime)) >=1 then 1 else 0 end as hour_from_intime

FROM mimic_icu.chartevents ce 
JOIN mimic_icu.d_items fl 
ON fl.itemid = ce.itemid

JOIN mimiciv.lookup lk 
ON lk.vital_name = fl.abbreviation
JOIN mimiciv.final_stays tbi ON
ce.stay_id = tbi.stay_id
LEFT JOIN  mimic_icu.icustays icu
ON tbi.stay_id = icu.stay_id

WHERE (lk.vital_name IN ('ABPd', 'ABPs', 'ABPm')) AND (icu.los >= 2);