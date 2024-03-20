#!/bin/bash

# to run this file:
# $ source $HOME/WS_Farid/base_essentials/puhti_intractive_cpu.sh

# Image_Retrieval_TUNI
# srun -J intrtv_cpuTEST --account=project_2004072 --partition=test --time=00-00:15:00 --mem=373G --ntasks=1 --cpus-per-task=40 --pty /bin/bash -i
# srun -J intrtv_cpu_large --account=project_2004072 --partition=large --time=00-08:00:00 --mem=310G --ntasks=1 --cpus-per-task=4 --pty /bin/bash -i
srun -J cpu_interactive --account=project_2004072 --partition=interactive --time=00-07:00:00 --mem=75G --ntasks=1 --cpus-per-task=2 --pty /bin/bash -i

module load git