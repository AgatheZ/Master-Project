WITH transformed AS
(
SELECT diagnosis,
case when diagnosis LIKE '%hemor%' then 'Hemorrhage'
else 'Other diagnosis' end as diag

FROM public.patient_info

)

SELECT diag, COUNT(1) FROM transformed
GROUP BY diag