PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)

wandb login $0

deepspeed --num_gpus=2 $PARENT_DIR/src/finetune_lora.py \
	--deepspeed $PARENT_DIR/src/deepspeed.json \
	--dataset promotion_v2 \
	--lora_model bloom_lora8 \
	--wandb_project bloom-lora \
	--wandb_run_name 0801-epoch
