#!/bin/bash

create_slurm_script () {

    WRITE_FILE="./brownian/${1}/${2}/${4}/brownian.sbatch"
    if test -f "$WRITE_FILE"; then 
        rm $WRITE_FILE
    fi
    
    printf "#!/bin/bash\n" >> $WRITE_FILE
    printf "#SBATCH --job-name=${1}_${2}_${4}_brownian\n" >> $WRITE_FILE
    printf "#SBATCH --time=24:00:00\n" >> $WRITE_FILE
    printf "#SBATCH --partition=hns\n" >> $WRITE_FILE
    printf "#SBATCH --nodes=1\n" >> $WRITE_FILE
    printf "#SBATCH -c 2\n" >> $WRITE_FILE
    printf "#SBATCH --mem-per-cpu=16G\n" >> $WRITE_FILE
    printf "#SBATCH --output=%%j_out.out\n" >> $WRITE_FILE
    printf "#SBATCH --error=%%j_err.err\n" >> $WRITE_FILE

    printf "\n\n" >> $WRITE_FILE
    printf "module purge \n" >> $WRITE_FILE
    printf "source /home/groups/rotskoff/shriramconda3/etc/profile.d/conda.sh \n" >> $WRITE_FILE
    
    printf "conda activate py38cpu \n" >> $WRITE_FILE

    printf "python $(pwd)/train.py --t_h ${2} --t_c 0.1 --var_h 0.1 --var_l 0.01 --time_per_cycle ${1} --dt ${3} --interval_length ${4} --num_intervals ${5}\n" >> $WRITE_FILE

}
mkdir brownian

# time_per_cycle=0.1
# t_h=1.
# dt=0.0002
# interval_length=1
# num_intervals=500
# mkdir ./brownian/${time_per_cycle}/${t_h}
# create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
# { cd ./brownian/${time_per_cycle}/${t_h}; pwd; sbatch brownian.sbatch; } &

time_per_cycle=0.1
t_h=1.
dt=0.0002
interval_length=5
num_intervals=100
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=1.0
t_h=1.
dt=0.0002
interval_length=5
num_intervals=1000
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &

time_per_cycle=1.0
t_h=1.
dt=0.0002
interval_length=10
num_intervals=500
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &




time_per_cycle=5.0
t_h=1.
dt=0.001
interval_length=5
num_intervals=1000
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=5.0
t_h=1.
dt=0.001
interval_length=10
num_intervals=500
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &




time_per_cycle=10.0
t_h=1.
dt=0.001
interval_length=5
num_intervals=2000
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=10.0
t_h=1.
dt=0.001
interval_length=10
num_intervals=1000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &



time_per_cycle=25.0
t_h=1.
dt=0.001
interval_length=5
num_intervals=5000
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=25.0
t_h=1.
dt=0.001
interval_length=10
num_intervals=2500
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &

time_per_cycle=25.0
t_h=1.
dt=0.001
interval_length=25
num_intervals=1000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &



time_per_cycle=50.0
t_h=1.
dt=0.001
interval_length=5
num_intervals=10000
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=50.0
t_h=1.
dt=0.001
interval_length=10
num_intervals=5000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=50.0
t_h=1.
dt=0.001
interval_length=50
num_intervals=1000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &

time_per_cycle=100.0
t_h=1.
dt=0.001
interval_length=5
num_intervals=20000
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=100.0
t_h=1.
dt=0.001
interval_length=10
num_intervals=10000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &

time_per_cycle=100.0
t_h=1.
dt=0.001
interval_length=100
num_intervals=1000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=500.0
t_h=1.
dt=0.001
interval_length=5
num_intervals=100000
mkdir ./brownian/${time_per_cycle}/
mkdir ./brownian/${time_per_cycle}/${t_h}
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=500.0
t_h=1.
dt=0.001
interval_length=10
num_intervals=50000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &


time_per_cycle=500.0
t_h=1.
dt=0.001
interval_length=500
num_intervals=1000
mkdir ./brownian/${time_per_cycle}/${t_h}/${interval_length}
create_slurm_script $time_per_cycle $t_h $dt $interval_length $num_intervals
{ cd ./brownian/${time_per_cycle}/${t_h}/${interval_length}; pwd; sbatch brownian.sbatch; } &