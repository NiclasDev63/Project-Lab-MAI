#!/bin/bash
#SBATCH -e /work/scratch/kurse/kurs00079/data/vox2/out/test.err
#SBATCH -o /work/scratch/kurse/kurs00079/data/vox2/out/test.out
#
# CPU specification
#SBATCH -n 1 # 1 process
#SBATCH -c 4 # 4 CPU cores per process
#SBATCH --mem-per-cpu=3800 # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 144:10:00 # in hours:minutes, or '#SBATCH -t 10' - just minutes
#SBATCH -A kurs00079
#SBATCH -p kurs00079
#SBATCH --reservation=kurs00079
# GPU specification
#SBATCH --gres=gpu:v100:1 # 1 GPU

module load gcc/8 python/3.9
cd /work/scratch/kurse/kurs00079/data/vox2/
source shared_env/shared_env3.9/bin/activate
cd shared_code/Project-Lab-MAI
python train_crossmodal.py

exit $EXITCODE
