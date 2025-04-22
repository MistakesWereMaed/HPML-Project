#!/bin/bash

# === Parse command-line arguments ===
while getopts "p:m:e:" opt; do
  case ${opt} in
    p )
      NUM_PROCS=$OPTARG
      ;;
    m )
      MODEL_TYPE=$OPTARG
      ;;
    e )
      EPOCHS=$OPTARG
      ;;
    \? )
      echo "Usage: $0 [-p num_procs] [-m model_type] [-e epochs]"
      exit 1
      ;;
  esac
done

# === Activate conda environment ===
source ~/.bashrc
conda activate torch
ml openmpi/5.0.5

# === MPI launch ===
mpirun --oversubscribe -np $NUM_PROCS \
    -x OMP_NUM_THREADS=1 \
    -x OMPI_MCA_pml=ob1 \
    -x OMPI_MCA_btl=self,vader,tcp \
    -x MASTER_ADDR=localhost \
    -x MASTER_PORT=12345 \
    python model_trainer.py --model $MODEL_TYPE --epochs $EPOCHS

