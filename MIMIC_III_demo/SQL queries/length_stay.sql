WITH transformed AS
(
SELECT length_stayicu
, case when length_stayicu < 1 then '< 24h'
  when  length_stayicu < 5 then '1 - 5'
  when  length_stayicu > 5 then '5 - 10'
  when  length_stayicu > 10 then '> 10'
else 'other' end as length_stay

FROM public.patient_info

)

SELECT length_stay, COUNT(1) FROM transformed
GROUP BY length_stay




