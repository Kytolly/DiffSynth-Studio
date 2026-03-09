CUDA_VISIBLE_DEVICES=2,3 accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path FollowBench/train \
  --dataset_metadata_path FollowBench/train/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 80 \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --save_steps 500 \
  --validation_steps 5 \
  --validation_prompt "Transform it into the third-person perspective." \
  --validation_video "FollowBench/train/train_case_00000/ego.mp4" \
  --validation_ref_image "FollowBench/train/train_case_00000/ref_img.jpg" \
  # --lora_checkpoint \
  
  