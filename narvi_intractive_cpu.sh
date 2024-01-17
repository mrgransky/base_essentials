#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

srun -J cpu_intractive  --partition=normal --time=00-09:59:59 --mem=120G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i