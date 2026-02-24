import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch


class LightningDataset(torch.utils.data.Dataset):
    def __init__(self, data_map, folder_path):
        self.data_map = data_map
        self.folder_path = folder_path
        self.timestamps = list(data_map.keys())

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        ts = self.timestamps[idx]
        files = self.data_map[ts]

        # get path of ki & total_prec & cape's image path
        x_atm_path = os.path.join(self.folder_path, files["x_paths"][0])
        ki, cape, precip = get_aligned_tensors(x_atm_path)

        # get path of entln & ds & lpi's image path
        y_file_path = os.path.join(self.folder_path, files["y_path"])
        entln, ds, lpi = get_aligned_tensors(y_file_path)

        # rearange the tensors to group atm' parameters together, ignoring ds
        x_tensor = torch.cat([ki, cape, precip, lpi], dim=0)

        # entln is the y_tensor
        y_tensor = entln

        return x_tensor, y_tensor


def organize_lightning_data(folder_path):
    """
    Get all the case's images in a given folder, and construct a dictionary where the key is the timestep/interval and the value is the image's path

    Args:
        folder_path: str, the path to the given images.
    """

    # dict to containt all atm' parameters and their paths
    data_map = {}

    for filename in os.listdir(folder_path):
        # extract date and hour from image name
        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})", filename)
        if not match:
            continue

        timestamp = match.group(1)

        if timestamp not in data_map:
            data_map[timestamp] = {"x_paths": [], "y_path": None}

        # identify the image layout (there are two types)
        if "ENTLN" in filename in filename:
            data_map[timestamp]["y_path"] = filename
        else:
            data_map[timestamp]["x_paths"].append(filename)

    # arrange the dict in a chronological order
    sorted_timestamps = sorted(data_map.keys())
    sorted_data_map = {ts: data_map[ts] for ts in sorted_timestamps}

    return sorted_data_map


def get_aligned_tensors(image_path, target_size=(256, 256)):
    """
    Get an image path, extract each subplot inside the given image, and transform each subplot to a gray scaled and resized tensor.

    Args:
        image_path: str, the path to a given image.
        target_size: tuple, the desired tensor size.
    """

    img = Image.open(image_path)
    width, height = img.size
    step = width // 3
    top_start = 200

    transform_pipeline = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize(target_size),
        ]
    )

    tensors = []
    for i in range(3):
        left = i * step
        right = (i + 1) * step
        cropped = img.crop((left, top_start, right, height))

        t_part = transform_pipeline(cropped)
        t_part = clear_map_artifacts(t_part)

        tensors.append(t_part)

    return (
        tensors  # returns list of tensors in this format- [tensor1, tensor2, tensor3]
    )


######### Helper functions #########


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def clear_map_artifacts(tensor):
    white_mask = tensor.mean(dim=0) > 0.95
    tensor[:, white_mask] = 0

    width = tensor.shape[2]
    colorbar_start = int(width * 0.9)
    tensor[:, :, colorbar_start:] = 0
    return tensor


def show_results(tensor, titles=None):
    if titles is None:
        titles = ["Layer 0", "Layer 1", "Layer 2"]

    fig, axes = plt.subplots(1, 12, figsize=(15, 5))
    print("TENSOR SHAPE", tensor.shape[0])
    for i in range(tensor.shape[0]):
        layer = tensor[i].numpy()
        axes[i].imshow(layer, cmap="jet")
        # axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


###################################

if __name__ == "__main__":
    data_folder = "/Users/karinpitlik/Desktop/DataScience/Thesis/Case1_Nov_2022_23_25/Ens/Graphs/UNITED/4by4/3_hours/jpeg"
    data_map = organize_lightning_data(data_folder)

    dataset = LightningDataset(data_map=data_map, folder_path=data_folder)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    for x_batch, y_batch in train_loader:
        for tensor in x_batch:
            pass
            show_results(tensor, ["ki", "cape", "precip", "lpi"])
