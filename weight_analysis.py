'''
Analyze neural network weights
'''
import h5py


def main():
    network_weights = \
        "output/classification/network_analysis/weights__holdout.hdf5"
    hdfile = h5py.File(network_weights, "r")
    for key in hdfile:
        for subkey in hdfile[key]:
            layer = hdfile[key][subkey]
            for sskey in hdfile[key][subkey]:
                data = layer[sskey]
                print(sskey)
                print(data[:])


if __name__ == '__main__':
    main()
