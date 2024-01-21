#!/bin/bash
# path to the Llama model 
MODEL=${1}

# what calibaration dataset to use
CALIB_DATA=wikitext2

bit=4

# arguments for different quantization settings
cmd_base="--wbits ${bit} --abits ${bit} --a_sym --w_sym --eval_ppl"
cmd_group="--act_group_size 128 --weight_group_size 128 --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian"
cmd_reorder_cache="--reorder --act_sort_metric hessian --cache_index"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_keep_fp16="--keeper 128 --keeper_precision 0"
cmd_keep_int8="--keeper 128 --keeper_precision 3"
cmd_adv="--use_gptq"
cmd_kv="--kv_cache"

index_arr=("0" "1" "2" "3" "4" "5" "6")

# commands for running the ablation study 
cmd_arr=(
    "${cmd_base}"
    "${cmd_base} ${cmd_reorder} ${cmd_keep_fp16}"
    "${cmd_base} ${cmd_reorder} ${cmd_keep_int8}"
    "${cmd_base} ${cmd_reorder_cache} ${cmd_keep_int8} ${cmd_group}"
    "${cmd_base} ${cmd_reorder_cache} ${cmd_keep_int8} ${cmd_group} ${cmd_clip}"
    "${cmd_base} ${cmd_reorder_cache} ${cmd_keep_int8} ${cmd_group} ${cmd_clip} ${cmd_adv}"
    "${cmd_base} ${cmd_reorder_cache} ${cmd_keep_int8} ${cmd_group} ${cmd_clip} ${cmd_adv} ${cmd_kv}"
)

log_arr=(
    "base"
    "keep-fp16"
    "keep-int8"
    "group"
    "clip"
    "gptq"
    "kv-int4"
)

dir=$(pwd)
resultFile=$dir/atom_w${bit}a${bit}_ablation_v3.csv
echo "method,wiki2" >> ${resultFile}

for idx in "${index_arr[@]}"
do
    logFile=$dir/atom_${model}_w${bit}a${bit}_${log_arr[${idx}]}.log

    echo "cmd config: " ${cmd_arr[${idx}]}
    python ${dir}/model/main.py ${MODEL} ${CALIB_DATA} ${cmd_arr[${idx}]} 2>&1 | tee ${logFile}

    wiki2=`cat $logFile | grep ",wikitext2," | awk -F ',' 'BEGIN { OFS = "," } {print $3}'`

    echo ${log_arr[${idx}]},${wiki2} 
    echo ${log_arr[${idx}]},${wiki2} >> ${resultFile}
    rm -f $logFile
done