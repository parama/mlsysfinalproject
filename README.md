# 6.887 Final Project

### Download and Installation 

Clone repository:
```
git clone --recurse-submodules git@github.com:parama/mlsysfinalproject.git
```

Download SOSD data:
```
cd SOSD
./scripts/download.sh
```

### Data Folder Structure

Example data layout 
```
project/
├── scr/
└── data/
    ├── wiki_ts_200M_uint64
    └── workloads/
	    └── wiki_ts_200M_uint64_workload100k_alpha1.1
```

### Build Learned Index

To compile the code, run 
```bash
sh build.sh
```
to compile the executables to the `build` directory.


### Benchmark

Run `./benchmark` to benchmark a lookup workloads.

---

### Running SOSD

To run SOSD on macOS, we can use the version released under the tag `mlforsys19`.

After running it locally, we got the following benchmark table

|               |       RMI |        RS |       ART |      FAST |       RBS |    B-tree |        BS |       TIP |        IS |
| ------------- | ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:|
| amzn32        |       307 |      1222 |       n/a |       602 |       376 |       674 |      1100 |       824 |      5021 |
| face32        |       338 |      1018 |      1478 |       515 |       339 |       566 |       954 |      1480 |      1235 |
| logn32        |       116 |       194 |       n/a |       561 |       553 |       509 |       880 |       770 |       n/a |
| norm32        |      92.2 |       604 |      1596 |       337 |       382 |       562 |       944 |      1472 |     11482 |
| uden32        |      69.7 |       568 |       138 |       486 |       373 |       526 |       815 |       191 |      55.5 |
| uspr32        |       202 |       456 |       n/a |       469 |       319 |       602 |       847 |       499 |       499 |
| amzn64        |       979 |      1689 |       n/a |       n/a |       866 |       727 |      1013 |       897 |      5866 |
| face64        |       981 |      1816 |     10702 |       n/a |       424 |       700 |      1758 |      1317 |      2000 |
| logn64        |       662 |      1103 |       386 |       n/a |       879 |       647 |      1133 |      1234 |       n/a |
| norm64        |       105 |       813 |      1856 |       n/a |       434 |       587 |      1719 |      1219 |     13017 |
| osmc64        |      1238 |      1352 |       n/a |       n/a |       577 |       732 |       956 |      7245 |    106801 |
| uden64        |      72.1 |       664 |       147 |       n/a |       403 |       575 |       923 |       223 |      77.8 |
| uspr64        |       243 |      1374 |       345 |       n/a |       334 |       622 |      1323 |       506 |       628 |
| wiki64        |       774 |      1189 |       n/a |       n/a |       482 |       723 |       990 |      2089 |      9819 |
| **avg**           |       441 |      1004 |      2081 |       495 |       481 |       625 |      1096 |      1426 |     13041 |


#### Steps to Reproduce

**1. Setup**
```bash
git clone --depth 1 --branch mlforsys19 https://github.com/learnedsystems/SOSD.git

cd SOSD 
. ./scripts/setup_anywhere.sh
```

**2. Runs**
* `scripts/download.sh` downloads and stores required data from the Internet
* `scripts/build_rmis.sh` compiles and builds the RMIs for each dataset
* `scripts/prepare.sh` constructs query workloads and compiles the benchmark
* `scripts/execute.sh` executes the benchmark on each workload, storing the results in `results`
* `scripts/create_leaderboard.sh` gathers all results in `results` into an easy to read table

or to run all the steps use `reproduce.sh`