---CODE ADAPTED FROM https://opendata.stackexchange.com/questions/6397/is-the-patients-height-available

DROP TABLE mimiciv.heightweight;
CREATE TABLE mimiciv.heightweight
AS
WITH FirstVRawData AS
  (SELECT c.charttime,
    c.itemid,c.subject_id,c.stay_id,
    CASE
      WHEN c.itemid IN (226512, 226531)
      THEN 'WEIGHT'
      WHEN c.itemid IN (226707, 226730)
      THEN 'HEIGHT'
    END AS parameter,
    CASE
      WHEN c.itemid   IN (226531) ---lbs to kgs 
      THEN c.valuenum * 0.45359237
      WHEN c.itemid   IN (226707)
      THEN c.valuenum * 2.54 --inches to cm 
      ELSE c.valuenum
    END AS valuenum
  FROM mimic_icu.chartevents c
  WHERE c.valuenum   IS NOT NULL
  AND ( ( c.itemid  IN (226512, 226531, 226707, 226730 
    )
  AND c.valuenum <> 0 )
    ) )
  --)

  --select * from FirstVRawData
, SingleParameters AS (
  SELECT DISTINCT subject_id,
         stay_id,
         parameter,
         first_value(valuenum) over (partition BY subject_id, stay_id, parameter order by charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_valuenum,
         MIN(valuenum) over (partition BY subject_id, stay_id, parameter order by charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)         AS min_valuenum,
         MAX(valuenum) over (partition BY subject_id, stay_id, parameter order by charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)         AS max_valuenum
    FROM FirstVRawData



--   ORDER BY subject_id,
--            stay_id,
--            parameter
  )
--select * from SingleParameters
, PivotParameters AS (SELECT subject_id, stay_id,
    MAX(case when parameter = 'HEIGHT' then first_valuenum else NULL end) AS height_first,
    MAX(case when parameter =  'HEIGHT' then min_valuenum else NULL end)   AS height_min,
    MAX(case when parameter =  'HEIGHT' then max_valuenum else NULL end)   AS height_max,
    MAX(case when parameter =  'WEIGHT' then first_valuenum else NULL end) AS weight_first,
    MAX(case when parameter =  'WEIGHT' then min_valuenum else NULL end)   AS weight_min,
    MAX(case when parameter =  'WEIGHT' then max_valuenum else NULL end)   AS weight_max
  FROM SingleParameters
  GROUP BY subject_id,
    stay_id
  )
--select * from PivotParameters
SELECT f.stay_id,
  f.subject_id,
  ROUND( cast(f.height_first as numeric), 2) AS height_first,
  ROUND(cast(f.height_min as numeric),2) AS height_min,
  ROUND(cast(f.height_max as numeric),2) AS height_max,
  ROUND(cast(f.weight_first as numeric), 2) AS weight_first,
  ROUND(cast(f.weight_min as numeric), 2)   AS weight_min,
  ROUND(cast(f.weight_max as numeric), 2)   AS weight_max

FROM PivotParameters f
ORDER BY subject_id, stay_id;