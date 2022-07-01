select distinct *
from mimic_icu.d_items
where category = 'Alarm'
limit 10
