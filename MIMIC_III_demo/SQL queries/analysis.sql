SELECT age AS val, COUNT(1) AS age from public.patient_info
GROUP BY age
ORDER BY val;

SELECT diagnosis, count(1) from public.patient_info
GROUP BY diagnosis
HAVING count(1) > 1;

SELECT COUNT(1), gender FROM public.patient_info 
GROUP BY gender;

