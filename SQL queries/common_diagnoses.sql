SELECT diagnosis, count(1) from public.patient_info AS common_diagnoses
GROUP BY diagnosis
HAVING count(1) > 1