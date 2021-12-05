echo "generating"
python3 generate_workloads.py -d SOSD/data/wiki_ts_200M_uint64 -t unit64 -s 200000000 -o workloads -a 1.1
echo "generating"
python3 generate_workloads.py -d SOSD/data/wiki_ts_200M_uint64 -t unit64 -s 200000000 -o workloads -a 1.3
echo "generating"
python3 generate_workloads.py -d SOSD/data/wiki_ts_200M_uint64 -t unit64 -s 200000000 -o workloads -a 1.5
echo "generating"
python3 generate_workloads.py -d SOSD/data/wiki_ts_200M_uint64 -t unit64 -s 200000000 -o workloads -a 1.7
echo "generating"
python3 generate_workloads.py -d SOSD/data/wiki_ts_200M_uint64 -t unit64 -s 200000000 -o workloads -a 1.9
