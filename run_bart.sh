#!/bin/bash

#PBS -q a6500g10q@vm-pbs2
#PBS -l select=1:ncpus=1:mem=2gb:ngpus=1
#PBS -l walltime=00:25:00
#PBS -m n

cd $PBS_O_WORKDIR
echo "I run on node: `uname -n`"
echo "My working directory is: $PBS_O_WORKDIR"
echo "Assigned to me nodes are:"
cat $PBS_NODEFILE
source /opt/shared/anaconda/anaconda3-2020/bin/activate
conda activate dnvaulin
python3 main.py
