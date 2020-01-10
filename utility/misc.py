# some more or less useful function usually for the preprocessing


def shadow_neigh(batch):
    """
    replace -1 value by the indices of shadow neighbors
    it is better to find an other solution: it is rather slow
    """
    list_max = []
    for i in range(len(batch.list_neigh)):
        batch.list_neigh[i][batch.list_neigh[i] < 0] = len(batch.list_neigh[i])
        list_max.append(len(batch.list_neigh[i]))

    for i in range(len(batch.list_pool)):
        batch.list_pool[i][batch.list_pool[i] < 0] = list_max[i]
    for i in range(len(batch.list_upsample)):
        batch.list_upsample[i][batch.list_upsample[i] < 0] = list_max[i+1]
    return batch



def get_list_constants(v, density, architecture, fdim, input_size, num_classes):
    """
    get the constants for the data preprocessing (size of radius, voxel size)
    and the network (radius of kpconv )
    inputs:
    - v: size of voxel
    - density : density
    - architecture: name of all layers
    """

    r = density * v
    list_voxel_size = [2*v, 4*v, 8*v, 16*v]
    list_radius = [r, 2*r, 4*r, 8*r, 16*r]

    r = density * 2 * v

    list_size = []
    list_radius_conv = []
    for name in architecture:
        list_radius_conv.append(r)
        if('resnetb' in name):
            list_size.append((input_size, 2*fdim))

            input_size = 2*fdim
        else:
            list_size.append((input_size, fdim))
            input_size = fdim
        if('strided' in name):
            fdim = 2*fdim
            r = 2*r
    list_size[-1] = (list_size[-2][1], num_classes)

    return architecture, list_voxel_size, list_radius, list_radius_conv, list_size
