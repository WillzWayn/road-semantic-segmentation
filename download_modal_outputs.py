import argparse
import subprocess
from pathlib import Path


def run(command):
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def volume_exists(name):
    code, _, err = run(["modal", "volume", "ls", name, "/"])
    if code == 0:
        return True
    if "not found" in err.lower():
        return False
    raise RuntimeError(err)


def download_volume(volume, remote_path, local_path, force=False):
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", volume, remote_path, str(local_path)]
    if force:
        cmd.insert(3, "--force")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download training outputs from Modal volumes")
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Local outputs root directory (default: outputs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local files/directories when downloading",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    checkpoints_dir = outputs_dir / "checkpoints"
    unet_dir = outputs_dir / "unet"

    if volume_exists("unet-outputs"):
        print("Found unified volume: unet-outputs")
        download_volume("unet-outputs", "/", outputs_dir, force=args.force)
        print(f"Downloaded to {outputs_dir}")
        return

    print("Unified volume not found. Falling back to legacy volumes...")
    found_any = False

    if volume_exists("unet-checkpoints"):
        download_volume("unet-checkpoints", "/", checkpoints_dir, force=args.force)
        print(f"Downloaded checkpoints to {checkpoints_dir}")
        found_any = True

    if volume_exists("unet-logs"):
        download_volume("unet-logs", "/", unet_dir, force=args.force)
        print(f"Downloaded logs/artifacts to {unet_dir}")
        found_any = True

    if not found_any:
        raise RuntimeError(
            "No known output volume found. Expected 'unet-outputs' or legacy 'unet-checkpoints'/'unet-logs'."
        )


if __name__ == "__main__":
    main()
