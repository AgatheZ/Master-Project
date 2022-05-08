DROP TABLE IF EXISTS mimiciiid.lookup CASCADE;
CREATE TABLE mimiciiid.lookup (final varchar(50), original varchar(50), item_code varchar(7), units varchar(10));

COPY mimiciiid.lookup(final, original, item_code, units) FROM 'lookup_table.csv' delimiter ',' CSV HEADER;

DROP TABLE IF EXISTS mimiciiid.vital_range CASCADE;
CREATE TABLE mimiciiid.vital_range (vital varchar(50), low integer, high integer);

COPY mimiciiid.vital_range(vital, low, high) FROM 'vitals_range.csv' delimiter ',' CSV HEADER;