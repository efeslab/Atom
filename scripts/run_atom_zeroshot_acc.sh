#!/bin/bash
# path to the Llama model 
MODEL=${1}

# what calibaration dataset to use
CALIB_DATA=wikitext2

BIT=4

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size 128 --weight_group_size 128 --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 128 --keeper_precision 3 --kv_cache --use_gptq"
cmd_eval="--eval_common_sense --lm_eval_limit -1 --multigpu"

dir=$(pwd)
resultFile=$dir/atom_llama_zeroshot_acc.csv

logFile=$dir/${MODEL}_w${BIT}a${BIT}_zeroshot.log
touch $logFile

CUDA_VISIBLE_DEVICE=0 python ${dir}/model/main.py ${MODEL} ${CALIB_DATA} \
    ${cmd_base} ${cmd_group} ${cmd_reorder} ${cmd_clip} ${cmd_adv} ${cmd_eval} \
    2>&1 | tee ${logFile}

# parse zero shot results
piqa=`cat $logFile | grep "INFO piqa :" | awk -F ':' 'BEGIN { OFS = "," } {print $2}'`
arc_easy=`cat $logFile | grep "INFO arc_easy :" | awk -F ':' 'BEGIN { OFS = "," } {print $2}'`
arc_challenge=`cat $logFile | grep "INFO arc_challenge :" | awk -F ':' 'BEGIN { OFS = "," } {print $2}'`
boolq=`cat $logFile | grep "INFO boolq :" | awk -F ':' 'BEGIN { OFS = "," } {print $2}'`
hellaswag=`cat $logFile | grep "INFO hellaswag :" | awk -F ':' 'BEGIN { OFS = "," } {print $2}'`
winogrande=`cat $logFile | grep "INFO winogrande :" | awk -F ':' 'BEGIN { OFS = "," } {print $2}'`

echo "model,bit,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"
echo ${MODEL},${BIT},${piqa},${arc_easy},${arc_challenge},${boolq},${hellaswag},${winogrande}
echo "model,bit,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande" >> ${resultFile}
echo ${MODEL},${BIT},${piqa},${arc_easy},${arc_challenge},${boolq},${hellaswag},${winogrande} >> ${resultFile}

rm $logFile