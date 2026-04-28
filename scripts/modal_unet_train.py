import modal

app = modal.App("unet-training")

dataset_volume = modal.Volume.from_name("unet-dataset", create_if_missing=True)
outputs_volume = modal.Volume.from_name("unet-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime")
    .pip_install("matplotlib", "numpy", "pillow", "torchvision", "tqdm")
    .add_local_dir("src", remote_path="/root/project/src")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    timeout=60 * 60 * 8,
    volumes={
        "/root/project/dataset": dataset_volume,
        "/root/project/outputs": outputs_volume,
    },
)
def train_unet(image_size: int = 256, batch_size: int = 8):
    import os
    import sys
    import runpy

    import torch
    from torch.utils import data as torch_data

    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    class FastDataLoader(torch_data.DataLoader):
        def __init__(self, *args, **kwargs):
            if kwargs.get("num_workers", 0) == 0:
                kwargs["num_workers"] = 8

            kwargs.setdefault("pin_memory", True)

            if kwargs["num_workers"] > 0:
                kwargs.setdefault("persistent_workers", True)
                kwargs.setdefault("prefetch_factor", 4)

            super().__init__(*args, **kwargs)

    torch_data.DataLoader = FastDataLoader

    os.chdir("/root/project")
    
    # Overwrite sys.argv so train_unet.py parses it correctly
    sys.argv = ["train_unet.py", "--image-size", str(image_size), "--batch-size", str(batch_size)]

    runpy.run_path("src/training/train_unet.py", run_name="__main__")


@app.local_entrypoint()
def main(image_size: int = 256, batch_size: int = 8):
    print(f"Starting remote training with size {image_size} and batch size {batch_size}...")
    train_unet.remote(image_size, batch_size)
