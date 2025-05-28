#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

dataset_id=2        # ▲
    if [[ $dataset_id == 1 ]]; then
        dataset="fmnist"; rounds=100; model="cnn"; ip="0.0.0.0:111";
        part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")    
    elif [[ $dataset_id == 2 ]]; then
        dataset="svhn"; rounds=100; model="cnn"; ip="0.0.0.0:112";
        part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")    
    elif [[ $dataset_id == 3 ]]; then
        dataset="cifar10"; rounds=200; model="cnn"; ip="0.0.0.0:113";
        part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")
    elif [[ $dataset_id == 4 ]]; then
        dataset="cifar100"; rounds=200; model="cnn"; ip="0.0.0.0:114";
        part_strategy_list=("iid" "labeldir0.1" "labelcnt0.1")
    else
        echo "wrong dataset."
        exit 1
    fi
    # dataset="fmnist"; rounds=100; model="cnn"; 
    lr=0.1; momentum=0; l2=0; epochs=3; batch_size=64; save_round=0; num_per_round=10; num_client=100; val_ratio=0.0; part_strategy="iid"; 
#######################################################################################

com_type="fedavg";      # √
ip="${ip}10"    # √
gpu=3
gpu_clients=(3 3 3 3 3 3 3 3 3 3) 

for c in 0; do  # lr
lrs=(0.03)
    lr=${lrs[${c}]}
for b in 0 1 2; do  # part
    part_strategy=${part_strategy_list[${b}]}
    
dir="../log/${dataset}+${num_client}/${dataset}+${part_strategy}/${model}+${com_type}+${lr}/"
        python ../train/server.py \
            --com_type ${com_type} \
            --model ${model} --dataset ${dataset} --lr ${lr} --momentum ${momentum} --l2 ${l2} \
            --rounds ${rounds} --epochs ${epochs} --batch_size ${batch_size} --save_round ${save_round} \
            --num_per_round ${num_per_round} --num_client ${num_client} \
            --gpu ${gpu} --ip ${ip} --log_dir ${dir} &

        for a in $(seq 0 $(($num_per_round-1))); do # client
        python ../train/client.py \
            --com_type ${com_type} \
            --model ${model} --dataset ${dataset} --part_strategy ${part_strategy} --num_client ${num_client} --id ${a} --val_ratio ${val_ratio} \
            --gpu ${gpu_clients[${a}]} --ip ${ip} --log_dir ${dir} &
        done
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM 
        wait
done
done
