export MODEL_NAME="./model/pretrained1.4"
export INSTANCE_DIR="./data/attack/dreambooth/3"
export OUTPUT_DIR="./dreambooth/v1.4/3"

python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of happy house" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of happy house in the box" \
  --validation_epochs=50 \
  --seed="0" \
  --mixed_precision="fp16"