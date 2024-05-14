export SAVE_IMAGE=0
export USE_AdaCaAttn=1
export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=3
torchrun --nproc_per_node 1 adapter_kv_cache/models/llama/example_chat_completion.py \
    --ckpt_dir "/app/llm_models/llama-2-7b-torch" \
    --tokenizer_path  "/app/llm_models/llama-2-7b-torch/tokenizer.model" \
    --max_seq_len 512 --max_batch_size 6