SELECT diagnosis, count(1) from public.patient_info
GROUP BY diagnosis
HAVING count(1) > 1