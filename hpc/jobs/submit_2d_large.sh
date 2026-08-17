#!/bin/bash
#PBS -N qpde_2D_large
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -o results/2Dhpc_run/pbs_stdout_large.log
#PBS -e results/2Dhpc_run/pbs_stderr_large.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

cd $PBS_O_WORKDIR
source ~/venvs/qpde/bin/activate

# Allow dirty tree (so we don't crash on pre-flight)
export PREFLIGHT_ALLOW_DIRTY=1

# Run ONLY Phase 2 (N=128, 256 for QSVT) and append to existing results
python3 hpc/runners/run_2d.py \
    --scheme fmg \
    --max-workers 4 \
    --n-values 128,256 \
    --solvers qsvt \
    --phase-tag large \
    --append \
    -I qsvt.max_degree=500 \
    -S max_wall_s=21600
