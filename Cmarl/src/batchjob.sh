#!/bin/sh

#  assert correct run dir
run_dir="src"
if ! [ "$(basename $PWD)" = $run_dir ];
then
    echo -e "\033[0;31mScript must be submitted from the directory: $run_dir\033[0m"
    exit 1
fi

# create dir for logs
mkdir -p "logs/"

### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Cmarl
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now 
#BSUB -W 7:00
### -- request 5GB of system-memory --
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
##BSUB -u s191780@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion-- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/Cmarl%J.out 
#BSUB -e logs/Cmarl%J.err 
### -- end of LSF options --

# activate env
source bachelor-env/bin/activate

# load additional modules
module load cuda/11.4

# run scripts
python DQN.py --task train --files data/filenames/training.txt data/filenames/training_landmark.txt --model_name CommNet --landmarks 0 1 2 3 4 5 --multiscale --viz 0 --train_freq 50 --write
