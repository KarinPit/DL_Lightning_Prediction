import os
import re


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


if __name__ == "__main__":
    data_folder = "/Users/karinpitlik/Desktop/DataScience/Thesis/Case1_Nov_2022_23_25/Ens/Graphs/UNITED/4by4/3_hours/"
    data_map = organize_lightning_data(data_folder)
    print(data_map)
