
# instantiate the starting csv file with column names
cat ICSD_data/ICSD_perovskites_empty.csv > ICSD_data/ICSD_perovskites.csv

# loop through all ICSD collected text files and append to 
for file_ in ICSD_texts/*
do
    sed '1d' $file_ >> ICSD_data/ICSD_perovskites.csv
done