# config.py

config = {
    "learning_rate": 1e-3,
    "batch_size": 32, #32 for cat dog 
    "epochs_pretrain": 50,
    "epochs_finetune": 50,
    "mask_ratio": 0.75,
    "num_classes": 2, #2 for cat dog
    "embed_dim": 2048,
    "patch_size": 16, # 16 for cat dog
    "image_size":256,#256 for catedog
    "use_pretrained_model": False,
    "train_path": "539project/datasets/datasets/train",  # Path to the dog cat train directory
    "val_path": "539project/datasets/datasets/val",      # Path to the dog cat validation directory
    "test_path": "539project/datasets/datasets/test",    # Path to the dog cat test directory
    "output_dir": "539project/training_output/semi_output",  # Directory to save logs and checkpoints
    "root_dir":"539project/datasets"
}
