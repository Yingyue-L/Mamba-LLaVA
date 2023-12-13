#!/bin/bash
python -m llava.eval.model_vqa_science \
    --model-path checkpoints/llava-mamba-2.8b-pretrain-fp32 \
    --model-base /data/yingyueli/hub/mamba-2.8b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-pretrain-fp32.jsonl \
    --single-pred-prompt \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-pretrain-fp32.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-pretrain-fp32_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-pretrain-fp32_result.json

python -m llava.eval.model_vqa_science \
    --model-path /data/yingyueli/hub/llava-mamba-2.8b-fp32 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-fp32.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-fp32.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-fp32_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-mamba-2.8b-fp32_result.json
