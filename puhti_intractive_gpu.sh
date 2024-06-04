#!/bin/bash

# to run this file:
# $ bash puhti_intractive_gpu.sh
# OR ...
# $ source puhti_intractive_gpu.sh

# never change!!!!! JUST TO TEST
srun -J intrctv_gpuTEST --account=project_2009043 --partition=gputest --gres=gpu:v100:1 --time=0-00:15:00 --mem=373G --ntasks=1 --cpus-per-task=4 --pty /bin/bash -i

# changing is allowed:
# srun -J gpu_interactive --account=project_2009043 --partition=gpu --gres=gpu:v100:1 --time=0-03:15:00 --mem=88G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i

# works after exit and source:
module load git