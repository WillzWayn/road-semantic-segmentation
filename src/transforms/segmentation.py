from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_mask_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


def build_inference_transform(image_size):
    return build_image_transform(image_size)
