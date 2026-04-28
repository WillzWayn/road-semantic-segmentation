from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_mask_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


def build_inference_transform(image_size):
    return build_image_transform(image_size)
