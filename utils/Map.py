import numpy as np
import matplotlib.pyplot as plt
import torch


def meter_to_index(loc_meter, resolution):
    """
    This function realizes the funcion of transforming the location in reality to 
    the location in an index form, and the resolution indicates how accurate the
    location is.

    PAY ATTENTION: we don't need to transform the height of the location into 
        index forms, cuz we need the information more precisely.

    Args:
        loc_meter: location in the meter form (other units can also work).
        resolution: how accurate the index is, the smaller the resolution, the more
            accurate the indexes.

    Returns:
        loc_index: loaction in the index form.
    """
    loc_index = torch.round(loc_meter/resolution).int()
    return loc_index

def loc_pair_to_map(loc_u, loc_d, resolution, map_size, mode='height'):
    """
    This function realizes the function of transforming the location pair of a user
    and a UAV into a map with the shape of map_size. Each value indicates the height
    of line between them at the specific index.

    Args:
        loc_u: the location of the user in a reality form.
        loc_d: the location of the UAV in a reality form.
        resolution: how accurate the index is, the smaller the resolution, the more
            accurate the indexes.
        map_size: the size of the map we need to generate.
        mode: indicate the output mode, 'height' for height map and 'indication' for 
            only indication of the connection of them.

    Returns:
        Position_map: a 2D map with specific size of map_size.
    """
    Position_map = torch.zeros(map_size) # Initlialize the map
    if torch.abs(loc_u[:2]-loc_d[:2]).sum()<0.001:
        loc_line = meter_to_index(loc_u[:2], resolution)
        Position_map[loc_line[0], loc_line[1]] = torch.abs(loc_d[-1]-loc_u[-1])

        return Position_map


        
    Sum_map = torch.zeros(map_size) # Calculate how many times the height at the same location has been calculated.
    delta = loc_u - loc_d # Set the location of the Drone as the standard location.
    row_delta, col_delta, height_delta = delta

    row_num, col_num = \
            torch.abs(torch.round(row_delta/resolution)).int(), torch.abs(torch.round(col_delta/resolution)).int()

    for i in range(row_num):
        tangent = col_delta/row_delta
        row = row_delta/row_num*i + loc_d[0]
        col = tangent*row_delta/row_num*i + loc_d[1]
        height = loc_d[2] + height_delta/row_num*i

        loc_row_index = meter_to_index(torch.tensor([row, col], dtype=float), resolution)
        # print('row, loc_row_index',loc_row_index)
        Position_map[loc_row_index[0], loc_row_index[1]] += height if mode=='height' else 255
        Sum_map[loc_row_index[0], loc_row_index[1]] += 1

    # Calculate for the col direction
    for i in range(col_num):
        tangent = row_delta/col_delta
        col = col_delta/col_num*i + loc_d[1]
        row = tangent*col_delta/col_num*i + loc_d[0]
        height = loc_d[2] + height_delta/col_num*i

        loc_row_index = meter_to_index(torch.tensor([row, col], dtype=float), resolution)
        # print('col, loc_row_index',loc_row_index)
        Position_map[loc_row_index[0], loc_row_index[1]] += height if mode=='height' else 255
        Sum_map[loc_row_index[0], loc_row_index[1]] += 1

    
    # print(Position_map)
    Sum_map[Sum_map == 0] += 1
    Fraction = 1/Sum_map
    Position_map = Position_map*Fraction

    return Position_map


def locs_to_map(locs, opt):
    """
    To generate the Position Map tensor.
    """
    batch_size = locs.size(0)
    Position_maps = torch.zeros([batch_size, 1, opt.map_size[-2], opt.map_size[-1]])

    for i in range(batch_size):
        Position_maps[i, 0, :] = loc_pair_to_map(locs[i, :3].cpu(), locs[i, 3:6].cpu(), opt.resolution, opt.map_size[-2:])

    return Position_maps





if __name__ == '__main__':
    size = [101,101]
    loc_u = torch.tensor([80, 70, 0])
    loc_d = torch.tensor([20, 10, 10])
    # loc_u = np.zeros([100, 3])
    # loc_d = np.zeros([100, 3])
    print(loc_d.shape)
    resolution = 1

    Position_map = loc_pair_to_map(loc_u, loc_d, resolution, size).cpu()
    print(Position_map[None, None,].shape)
    print(Position_map)
    plt.imshow(Position_map)
    plt.show()
