import  torch

def average_between_rows(input_tensor):
    x, _ = input_tensor.shape
    half_x = x // 2

    # Reshape the input tensor to (x//2, 2, 256)
    reshaped_input = input_tensor[:2 * half_x].view(half_x, 2, -1)

    # Calculate the average between consecutive rows
    average_tensor = torch.mean(reshaped_input, dim=1)

    return average_tensor