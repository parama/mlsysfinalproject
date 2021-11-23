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

### Build Learned Index

To compile the code, run 
```bash
sh build.sh
```
to compile the executables to the `build` directory.


### Benchmark

Run `./build/benchmark_learned_index` to benchmark a lookup workloads.
