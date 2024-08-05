accelerate launch --config_file "configs/fsdp_config.yaml" train.py \
--seed 100 \
--model_name_or_path "meta-llama/Meta-Llama-3-8B" \
--dataset_name "Yukang/LongAlpaca-12k" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train" \
--max_seq_len 8192 \
--num_train_epochs 1 \
--logging_steps 5 \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--bf16 True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama3" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora False \
--use_4bit_quantization False

accelerate launch --config_file "configs/fsdp_config.yaml" train.py \
--seed 100 \
--model_name_or_path "meta-llama/Meta-Llama-3-8B" \
--dataset_name "Yukang/LongAlpaca-12k" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train" \
--max_seq_len 16384 \
--num_train_epochs 1 \
--logging_steps 5 \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--bf16 True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama3" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora False \
--use_4bit_quantization False


accelerate launch --config_file "configs/fsdp_config.yaml" train.py \
--seed 100 \
--model_name_or_path "meta-llama/Meta-Llama-3-8B" \
--dataset_name "Yukang/LongAlpaca-12k" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train" \
--max_seq_len 30000 \
--num_train_epochs 1 \
--logging_steps 5 \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--bf16 True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama3" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora False \
--use_4bit_quantization False