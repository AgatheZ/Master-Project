SELECT 
ce.icustay_id -- for each icustay_id
,ce.charttime -- time where the vital was taken
,di.label -- vital name
,ce.value AS vital_reading -- vital reading

FROM mimiciiid.chartevents ce
INNER JOIN mimiciiid.d_items di
ON ce.itemid = di.itemid

WHERE (di.LABEL) = 'ICP ventricle'
