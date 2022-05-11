import torch

def load_dict(mainnet, subnetPath, preffix):
    """
    This function realizes loading the state dict of subnet in the main net.

    Args:
        mainnet: the net in need of loading the subnet.
        subnetPath: the path of the subnet's pth file.
        pre_name: the subnet's pre_name in the mainnet.
    """
    preffix = preffix + '.'
    mainnet_stateDict = mainnet.state_dict()
    subnet_stateDict = torch.load(subnetPath)
    stateDict_forupdate = {preffix+k:v for k, v in subnet_stateDict.items() if preffix+k in mainnet_stateDict.keys()}
    # print(stateDict_forupdate)
    # UNet_dict = {k[start:]:v for k, v in model_dict.items() if k[start:] in map_dict.keys()}
    mainnet_stateDict.update(stateDict_forupdate)
    mainnet.load_state_dict(mainnet_stateDict)

    return mainnet

