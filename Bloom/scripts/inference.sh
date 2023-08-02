PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)

python $PARENT_DIR/src/inference.py \
	    --lora_model bloom_lora7 \
	        --dataset tw_title
