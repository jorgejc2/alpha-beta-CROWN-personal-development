import torch

def rename_keys(input_file, output_file, key_mapping):
    # Load the state dictionary from the input file
    state_dict = torch.load(input_file)
    print("entire state dict\n", state_dict.keys())
    print("new state dict\n", state_dict['state_dict'].keys())

    # Rename keys according to the provided mapping
    # for old_key, new_key in key_mapping.items():
    #     print(old_key, new_key)
    #     if old_key in state_dict:
    #         print(f'removing {old_key} from {state_dict[old_key]} and placing in {new_key}')
    #         state_dict[new_key] = state_dict.pop(old_key)

    # Save the modified state dictionary to the output file
    torch.save(state_dict, output_file)

# Example usage:
if __name__ == "__main__":
    input_file = "cifar_conv_small_sigmoid_backup.pth"
    output_file = "cifar_conv_small_sigmoid.pth"
    key_mapping = {
        "linear_masked_relu1.slope": "linear_masked_relu1.alpha",
        "linear_masked_relu2.slope": "linear_masked_relu2.alpha",
        "linear_masked_relu3.slope": "linear_masked_relu3.alpha",
    }
    print("starting swap")
    rename_keys(input_file, output_file, key_mapping)
