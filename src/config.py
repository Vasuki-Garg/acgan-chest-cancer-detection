import os

def default_paths():

    base = " "
    return {
        "base_dir": base,
        "filtered_dataset_dir": os.path.join(base, "Data/path_to_filtered_dataset"),
        "train_dir": os.path.join(base, "Data/path_to_filtered_dataset/train"),
        "val_dir": os.path.join(base, "Data/path_to_filtered_dataset/val"),
        "test_dir": os.path.join(base, "Data/path_to_filtered_dataset/test"),
        "model_dir": os.path.join(base, "saved_models"),
        "runs_dir": os.path.join(base, "runs"),
        "generated_dir": os.path.join(base, "generated_outputs"),
    }
