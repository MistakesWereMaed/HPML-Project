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
      echo "Usage: $0 [-p num_procs] [-m model_type] [-e epochs] [-c deepspeed_config.json]"
      exit 1
      ;;
  esac
done

# === Activate conda environment ===
source ~/.bashrc
conda activate torch
ml openmpi/5.0.5

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

# === DeepSpeed launch ===
deepspeed \
    --num_gpus=$NUM_PROCS \
    model_trainer.py \
    --model $MODEL_TYPE \
    --epochs $EPOCHS \
