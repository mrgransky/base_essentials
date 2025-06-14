#!/bin/bash

# to run this file:
# $ source $HOME/WS_Farid/base_essentials/puhti_intractive_cpu.sh

# Image_Retrieval_TUNI
# srun -J intrtv_cpuTEST --account=project_2009043 --partition=test --time=00-00:15:00 --mem=32G --ntasks=1 --cpus-per-task=40 --pty /bin/bash -i
# srun -J small_cpu --account=project_2009043 --partition=small --time=00-00:30:00 --mem=64G --ntasks=1 --cpus-per-task=4 --pty /bin/bash -i
# srun -J intrtv_cpu_hugemem --account=project_2009043 --partition=hugemem --time=00-10:00:00 --mem=473G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
srun -J cpu_interactive --account=project_2009043 --partition=interactive --time=00-07:00:00 --mem=75G --ntasks=1 --cpus-per-task=8 --pty /bin/bash -i

module load git