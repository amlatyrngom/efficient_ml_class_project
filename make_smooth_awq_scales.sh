smoothquant_path=smoothquant
# Get model short name from the first argument
model_short_name=$1
pt_file="$smoothquant_path/act_scales/$model_short_name-awq.pt"
if [ -f "$pt_file" ]; then
    echo "Act scales already exist for $model_short_name-awq"
    exit 0
fi
# Get huggingface model name from the short name
# Copilot: If startswith opt, then path is facebook/$model_short_name
# Copilot: If startswith llama, then path is meta-llama/$model_short_name-hf
if [[ $model_short_name == opt* ]]; then
    model_full_name="facebook/$model_short_name"
elif [[ $model_short_name == llama-2* ]]; then
    model_full_name="meta-llama/$model_short_name-hf"
elif [[ $model_short_name == llama-3* ]]; then
    model_full_name="meta-llama/$model_short_name"
else
    echo "Model short name must start with opt or llama-2 or llama-3"
    exit 1
fi
validation_data_url="https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
validation_data_path="$smoothquant_path/act_scales/val.jsonl.zst"
if [ ! -f "$validation_data_path" ]; then
    wget $validation_data_url -O $validation_data_path
fi

python $smoothquant_path/examples/generate_act_scales.py \
    --model-name "$model_full_name" \
    --output-path "$pt_file" \
    --dataset-path "$validation_data_path" \
    --awq
