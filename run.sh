ml gcc/8 python/3.10
source ../mai/bin/activate
# pip install -r requirements.txt
# pip install wandb
# python train_intra_loss.py --data_root ../../data/vox2
python adaface_eval.py
deactivate