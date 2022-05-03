SELECT age AS val, COUNT(1) AS age from public.patient_info
GROUP BY age
ORDER BY val