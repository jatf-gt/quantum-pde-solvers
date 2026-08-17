#!/bin/bash
#PBS -N qpde_3D_large
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -o results/3Dhpc_run/pbs_stdout_large.log
#PBS -e results/3Dhpc_run/pbs_stderr_large.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

cd $PBS_O_WORKDIR
source ~/venvs/qpde/bin/activate

# Allow dirty tree
export PREFLIGHT_ALLOW_DIRTY=1

# Run ONLY Phase 2 (N=32 for QSVT) and append to existing results
python3 hpc/runners/run_3d.py \
    --scheme fmg \
    --max-workers 4 \
    --n-values 32 \
    --solvers qsvt \
    --phase-tag large \
    --append \
    -I qsvt.max_degree=500 \
    -S max_wall_s=36000
