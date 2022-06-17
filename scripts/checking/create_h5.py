import argparse
import os
import numpy as np
import h5py

parser = argparse.ArgumentParser('Split.')
parser.add_argument('in_points', type=str, help='points')
parser.add_argument('--in_points_name', type=str, default='points_uniform500000.npz', help='points name')
parser.add_argument('--out_points_name', type=str, default='points_uniform500000.h5', help='new name')
parser.add_argument('--unpack_bits', action='store_true', help='unpack bits')
parser.add_argument('--chunk_size', type=int, default=2048, help='chunk size')

if __name__ == '__main__':
    args = parser.parse_args()
    input_filename = os.path.join(args.in_points, args.in_points_name)
    pack = np.load(input_filename)

    points = pack['points']
    occupancies = pack['occupancies']
    if args.unpack_bits:
        occupancies = np.unpackbits(occupancies)[:points.shape[0]]

    loc = pack['loc']
    scale = pack['scale']

    #input range
    dirname = args.in_points
    filename = os.path.join(dirname, args.out_points_name)

    h5file = h5py.File(filename, 'w')
    chunk_size = args.chunk_size
    h5file.create_dataset("points", data=points)
    h5file.create_dataset("occupancies", data=occupancies)
    h5file.create_dataset("loc", data=loc)
    h5file.create_dataset("scale", data=scale)
    h5file.close()

    print('Finished ', dirname )
