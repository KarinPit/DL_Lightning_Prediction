import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch


def organize_lightning_data(folder_path):
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


def get_aligned_tensors(image_path, is_entln=False):
    img = Image.open(image_path)
    width, height = img.size
    step = width // 3
    top_start = 200

    target_size = (256, 256)

    # הוספנו Grayscale כדי להפוך 3 ערוצים ל-1
    transform_pipeline = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize(target_size),
            torchvision.transforms.ToTensor(),
        ]
    )

    if is_entln:
        return transform_pipeline(img.crop((0, 0, step, height)))
    else:
        # כל 't' כאן יהיה עכשיו בגודל (1, 256, 256)
        t1 = clear_map_artifacts(
            transform_pipeline(img.crop((0, top_start, step, height)))
        )
        t2 = clear_map_artifacts(
            transform_pipeline(img.crop((step, top_start, 2 * step, height)))
        )
        t3 = clear_map_artifacts(
            transform_pipeline(img.crop((2 * step, top_start, width, height)))
        )

        # האיחוד ייתן לנו בדיוק (3, 256, 256)
        return torch.cat([t1, t2, t3], dim=0)


######### Helper functions #########


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def clear_map_artifacts(tensor):
    mask = tensor > 0.95
    tensor[mask] = 0

    width = tensor.shape[2]
    colorbar_start = int(width * 0.9)
    tensor[:, :, colorbar_start:] = 0
    return tensor


def show_cropped_tensors():
    full_tensor = get_aligned_tensors(
        "/Users/karinpitlik/Desktop/DataScience/Thesis/Case1_Nov_2022_23_25/Ens/Graphs/UNITED/4by4/3_hours/jpeg/2022-11-24_03_00_00_to_2022-11-24_06_00_00_KI, CAPE and Total Precipitation.jpg"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["KI (Layer 0)", "CAPE (Layer 1)", "Precip (Layer 2)"]

    for i in range(3):
        layer = full_tensor[i].numpy()
        axes[i].imshow(layer, cmap="jet")
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


###################################


if __name__ == "__main__":
    data_folder = "/Users/karinpitlik/Desktop/DataScience/Thesis/Case1_Nov_2022_23_25/Ens/Graphs/UNITED/4by4/3_hours/jpeg"
    data_map = organize_lightning_data(data_folder)

    cropped_img = get_aligned_tensors(
        "/Users/karinpitlik/Desktop/DataScience/Thesis/Case1_Nov_2022_23_25/Ens/Graphs/UNITED/4by4/3_hours/jpeg/2022-11-24_03_00_00_to_2022-11-24_06_00_00_KI, CAPE and Total Precipitation.jpg"
    )

    show_cropped_tensors()
