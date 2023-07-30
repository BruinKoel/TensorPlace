import os
import torch
def find_files_by_extension(paths, extension, search_depth=-1):
    if isinstance(paths, str):
        paths = [paths]

    file_list = []

    def search_files(current_path, current_depth):
        if search_depth >= 0 and current_depth > search_depth:
            return

        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)

            if os.path.isfile(item_path) and item.endswith(f".{extension}"):
                file_list.append(item_path)
            elif os.path.isdir(item_path):
                search_files(item_path, current_depth + 1)

    for path in paths:
        if os.path.isfile(path):
            file_list.append(path)
        elif os.path.isdir(path):
            search_files(path, 0)

    return file_list

def normalize_volume(tensor):
    loudness = torch.max(torch.abs(tensor))
    return tensor / loudness


def split_and_pad(tensor, x):
    # Get the current shape of the tensor (1, x)
    shape = tensor.shape

    # Calculate the remainder when n is divided by x
    rem = shape[1] % x

    # Calculate the number of zeros to add
    num_zeros = x - rem if rem != 0 else 0

    # Create a new tensor by concatenating zeros along the second dimension
    padded_tensor = torch.cat((tensor.squeeze(0), torch.zeros(num_zeros)), dim=0).unsqueeze(0)
    print("!")
    # Calculate the new value of n after padding
    n_with_padding = shape[1] + num_zeros

    # Reshape the tensor to (1, batches, x)
    reshaped_tensor = padded_tensor.view(1, n_with_padding//x, x)

    return reshaped_tensor