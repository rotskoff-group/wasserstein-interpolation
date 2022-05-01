#!/bin/bash

create_slurm_script () {

    WRITE_FILE="./brownian_prx_lr/${1}/${2}/${5}/${4}/brownian_prx_lr.sbatch"
    if test -f "$WRITE_FILE"; then 
        rm $WRITE_FILE
    fi
    
    printf "#!/bin/bash\n" >> $WRITE_FILE
    printf "#SBATCH --job-name=${1}_${2}_${5}_${4}_brownian_prx_lr\n" >> $WRITE_FILE
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

    printf "python $(pwd)/test_prx_lr.py --d ${4} --t_h ${2} --t_c 0.1 --var_h 0.1 --var_l 0.01 --time_per_cycle ${1} --dt ${3} --interval_length ${5} --num_intervals ${6}\n" >> $WRITE_FILE

}
mkdir brownian_prx_lr
t_h=1.
all_d=(1E-5 0.02 0.2 1)



time_per_cycle=0.1
dt=0.0002
interval_length=5
num_intervals=100
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}


for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done

time_per_cycle=1.0
dt=0.0002
interval_length=5
num_intervals=1000
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=1.0
dt=0.0002
interval_length=10
num_intervals=500
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=5.0
dt=0.001
interval_length=5
num_intervals=1000
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=5.0
dt=0.001
interval_length=10
num_intervals=500
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=10.0
dt=0.001
interval_length=5
num_intervals=2000
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=10.0
dt=0.001
interval_length=10
num_intervals=1000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=25.0
dt=0.001
interval_length=5
num_intervals=5000
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=25.0
dt=0.001
interval_length=10
num_intervals=2500
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done

time_per_cycle=25.0
dt=0.001
interval_length=25
num_intervals=1000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=50.0
dt=0.001
interval_length=5
num_intervals=10000
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=50.0
dt=0.001
interval_length=10
num_intervals=5000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done

time_per_cycle=50.0
dt=0.001
interval_length=50
num_intervals=1000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done

time_per_cycle=100.0
dt=0.001
interval_length=5
num_intervals=20000
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=100.0
dt=0.001
interval_length=10
num_intervals=10000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done

time_per_cycle=100.0
dt=0.001
interval_length=100
num_intervals=1000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done

time_per_cycle=500.0
dt=0.001
interval_length=5
num_intervals=100000
mkdir ./brownian_prx_lr/${time_per_cycle}/
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done


time_per_cycle=500.0
dt=0.001
interval_length=10
num_intervals=50000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done

time_per_cycle=500.0
dt=0.001
interval_length=500
num_intervals=1000
mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}
for d in "${all_d[@]}"
do
    mkdir ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/
    create_slurm_script $time_per_cycle $t_h $dt ${d} $interval_length $num_intervals
    { cd ./brownian_prx_lr/${time_per_cycle}/${t_h}/${interval_length}/${d}/; pwd; sbatch brownian_prx_lr.sbatch; } &
done