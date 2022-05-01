#!/bin/bash

create_slurm_script () {
    time_per_cycle=${1}
    interval_length=${2}
    reg=${3}


    WRITE_FILE="./qubit/${1}_${2}_${3}/qubit.sbatch"
    if test -f "$WRITE_FILE"; then 
        rm $WRITE_FILE
    fi
    
    printf "#!/bin/bash\n" >> $WRITE_FILE
    printf "#SBATCH --job-name=${1}_${2}_${3}_qubit\n" >> $WRITE_FILE
    printf "#SBATCH --time=24:00:00\n" >> $WRITE_FILE
    printf "#SBATCH --partition=hns\n" >> $WRITE_FILE
    printf "#SBATCH --nodes=1\n" >> $WRITE_FILE
    printf "#SBATCH -c 2\n" >> $WRITE_FILE
    printf "#SBATCH --mem-per-cpu=16G\n" >> $WRITE_FILE
    printf "#SBATCH --output=%%x_out.out\n" >> $WRITE_FILE
    printf "#SBATCH --error=%%x_err.err\n" >> $WRITE_FILE

    printf "\n\n" >> $WRITE_FILE
    printf "module purge \n" >> $WRITE_FILE
    printf "source /home/groups/rotskoff/shriramconda3/etc/profile.d/conda.sh \n" >> $WRITE_FILE
    
    printf "conda activate py38cpu \n" >> $WRITE_FILE

    printf "python $(pwd)/train.py --tau ${1} --interval_length ${2} --dt 0.001 --reg ${3} --num_intervals ${4} --clip_gradients \n" >> $WRITE_FILE

}

mkdir qubit


time_per_cycle=0.1
interval_length=1
num_intervals=100
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

time_per_cycle=0.1
interval_length=5
num_intervals=20
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &



time_per_cycle=0.2
interval_length=1
reg=0
num_intervals=200
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

time_per_cycle=0.2
interval_length=5
reg=0
num_intervals=40
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=0.3
interval_length=1
reg=0
num_intervals=300
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

time_per_cycle=0.3
interval_length=5
reg=0
num_intervals=60
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=0.4
interval_length=1
num_intervals=400
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

time_per_cycle=0.4
interval_length=5
num_intervals=80
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

time_per_cycle=0.5
interval_length=1
num_intervals=500
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=0.5
interval_length=5
num_intervals=100
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &




time_per_cycle=1
interval_length=1
num_intervals=1000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

time_per_cycle=1
interval_length=5
num_intervals=200
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=2
interval_length=1
num_intervals=2000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=2
interval_length=5
num_intervals=400
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=5
interval_length=1
num_intervals=5000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &



time_per_cycle=5
interval_length=5
num_intervals=1000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

time_per_cycle=10
interval_length=1
num_intervals=10000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=10
interval_length=5
num_intervals=2000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &




time_per_cycle=25
interval_length=1
num_intervals=25000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &


time_per_cycle=25
interval_length=5
num_intervals=5000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &



time_per_cycle=50
interval_length=1
num_intervals=50000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &



time_per_cycle=50
interval_length=5
num_intervals=10000
reg=0
mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
{ cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

# time_per_cycle=50
# interval_length=10
# num_intervals=5000
# reg=0
# mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
# create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
# { cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &

# time_per_cycle=50
# interval_length=50
# num_intervals=1000
# reg=0
# mkdir ./qubit/${time_per_cycle}_${interval_length}_${reg}
# create_slurm_script $time_per_cycle $interval_length $reg $num_intervals
# { cd ./qubit/${time_per_cycle}_${interval_length}_${reg}; pwd; sbatch qubit.sbatch; } &
