#!/bin/sh 

# bsub -Is -q gpu -m gpu01 -J test_zyh -n 8 -gpu "num=1" -o outoput_test.txt -e error.txt << EOF
# conda activate loretta
# /hpc/home/connect.yzuo099/loretta/bert_model/run_all_bert_exp.sh
# EOF


bsub -Is -q gpu -m gpu01 -J test_zyh -gpu "num=4" -o outoput_test.txt -e error.txt << EOF
source ~/.bashrc
conda activate loretta
/hpc/home/connect.yzuo099/loretta/bert_model/run_all_bert_exp_hpceda.sh
EOF