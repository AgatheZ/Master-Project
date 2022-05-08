SELECT age AS val, COUNT(1) AS age FROM public.patient_info
GROUP BY age
ORDER BY val;

SELECT diagnosis, COUNT(1) FROM public.patient_info
GROUP BY diagnosis
HAVING COUT(1) > 1;

SELECT COUNT(1), gender FROM public.patient_info 
GROUP BY gender;

