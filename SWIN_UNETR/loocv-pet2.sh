# python3 main.py \
#     --exp=pet-fold2 \
#     --data_dir=../best_data_normal/best_data_normal \
#     --json_list=fold2_pet.json \
#     --in_channels=1 \
#     --out_channels=1 \
#     --max_epochs=1000 \
#     --use_checkpoint \
#     --save_checkpoint \
#     --val_every=50 \
#     --batch_size=1 \
#     --sw_batch_size=1 \
#     --infer_overlap=0.85 \
#     --roi_x=160 \
#     --roi_y=160 \
#     --roi_z=256 \
#     --optim_lr=1e-3 \
#     --optim_name=sgd \
#     --workers=8 \
#     --gpu=0 \
# #   --pretrained_dir=pretrain1111/ \
# #   --pretrained_model_name=model_bestValRMSE.pt \

# wait

python3 main.py \
    --exp=pet-fold3 \
    --data_dir=../best_data_normal/best_data_normal \
    --json_list=fold3_pet.json \
    --in_channels=1 \
    --out_channels=1 \
    --max_epochs=1000 \
    --use_checkpoint \
    --save_checkpoint \
    --val_every=50 \
    --batch_size=1 \
    --sw_batch_size=1 \
    --infer_overlap=0.85 \
    --roi_x=160 \
    --roi_y=160 \
    --roi_z=256 \
    --optim_lr=1e-3 \
    --optim_name=sgd \
    --workers=8 \
    --gpu=0 \
#   --pretrained_dir=pretrain1111/ \
#   --pretrained_model_name=model_bestValRMSE.pt \