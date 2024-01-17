#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

# Image_Retrieval_TUNI
# srun -J intrtv_cpu --account=project_2004072 --partition=test --time=00-00:15:00 --mem=185G --ntasks=1 --cpus-per-task=16 --pty /bin/bash -i
srun -J intrtv_cpu --account=project_2004072 --partition=large --time=00-08:00:00 --mem=168G --ntasks=1 --cpus-per-task=4 --pty /bin/bash -i
# srun -J cpu_large_interactive --account=project_2004072 --partition=large --time=00-23:59:00 --mem=64G --ntasks=1 --cpus-per-task=2 --pty /bin/bash -i

module load git