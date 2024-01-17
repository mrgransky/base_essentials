#!/bin/bash
srun -J gpu_interactive  --partition=test --gres=gpu:teslav100:1 --time=00-04:00:00 --mem=16G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
#module load git
#module load pytorch