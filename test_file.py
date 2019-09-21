import os
import h5py
import numpy as np
from lazy_loading import DatasetView
import secrets
from random import shuffle
from numpy.testing import assert_array_equal

def main():

    f = h5py.File('testfile.hdf5')
    ndims = 4
    secret_rand = secrets.SystemRandom()
    dset = f.create_dataset(name='test_dataset', data=np.random.rand(*secret_rand.sample(range(1, 200//ndims), ndims)))
    dsetview = DatasetView(dset)

    randslice = lambda d: slice(*secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(1, dsetview.shape[d] + 1), 1))

    for _ in range(200):
        slice_list = [randslice(0),
                      np.s_[:],
                      np.index_exp[:],
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape)))]

        slice_list = [secrets.choice(slice_list) for i in range(5)]

        shuffle_list = [list(range(ndims)) for i in range(5)]
        for li in shuffle_list: shuffle(li)

        assert_array_equal(dset[slice_list[1]],  dsetview.LazySlice[slice_list[1]][:])

        assert_array_equal(dsetview[slice_list[1]][slice_list[2]],  
              dsetview.LazySlice[slice_list[1]].LazySlice[slice_list[2]][:])

        assert_array_equal(dsetview[slice_list[1]][slice_list[2]][slice_list[3]], 
              dsetview.LazySlice[slice_list[1]].LazySlice[slice_list[2]].LazySlice[slice_list[3]][:])

        assert_array_equal(dset[slice_list[1]][slice_list[2]][slice_list[3]],  
              dsetview.LazySlice[slice_list[1]].LazySlice[slice_list[2]].LazySlice[slice_list[3]][:])

        assert_array_equal(dset[slice_list[1]].transpose(), dsetview.LazySlice[slice_list[1]].LazyTranspose()[:])

        assert_array_equal(dsetview[slice_list[1]].transpose(), dsetview.LazySlice[slice_list[1]].LazyTranspose()[:])

        assert_array_equal(dsetview[slice_list[1]].transpose()[slice_list[2]], 
              dsetview.LazySlice[slice_list[1]].LazyTranspose().LazySlice[slice_list[2]][:])

        assert_array_equal(dsetview[slice_list[1]].transpose()[slice_list[2]][slice_list[3]], 
              dsetview.LazySlice[slice_list[1]].LazyTranspose().LazySlice[slice_list[2]].LazySlice[slice_list[3]][:])

        assert_array_equal(dset[slice_list[1]].transpose()[slice_list[2]][slice_list[3]], 
              dsetview.LazySlice[slice_list[1]].LazyTranspose().LazySlice[slice_list[2]].LazySlice[slice_list[3]][:])

        assert_array_equal( dsetview[slice_list[1]].transpose(shuffle_list[3])[slice_list[2]][slice_list[3]], 
              dsetview.LazySlice[slice_list[1]].LazyTranspose(shuffle_list[3]).LazySlice[slice_list[2]].LazySlice[slice_list[3]].dsetread())

        assert_array_equal( dset[slice_list[1]].transpose(shuffle_list[3])[slice_list[2]][slice_list[3]].transpose(shuffle_list[4]),
             dsetview.LazySlice[slice_list[1]].LazyTranspose(shuffle_list[3]).LazySlice[slice_list[2]]
             .LazySlice[slice_list[3]].LazyTranspose(shuffle_list[4])[:])

        assert_array_equal(dset[:].transpose(shuffle_list[0]), dsetview.LazyTranspose(shuffle_list[0]).dsetread()[:])

        assert_array_equal(dsetview.transpose(shuffle_list[0])[slice_list[1]].transpose(shuffle_list[1])[slice_list[2]][slice_list[3]]
             .transpose(shuffle_list[2])[slice_list[4]].transpose(),
             dsetview.LazyTranspose(shuffle_list[0]).LazySlice[slice_list[1]].LazyTranspose(shuffle_list[1]).
             LazySlice[slice_list[2]].LazySlice[slice_list[3]].LazyTranspose(shuffle_list[2]).LazySlice[slice_list[4]].LazyTranspose()[:])

        assert_array_equal(dset[:].transpose(shuffle_list[0])[slice_list[1]].transpose(shuffle_list[1])[slice_list[2]][slice_list[3]].
             transpose(shuffle_list[2])[slice_list[4]].transpose(),
             dsetview.LazyTranspose(shuffle_list[0]).LazySlice[slice_list[1]].LazyTranspose(shuffle_list[1])
             .LazySlice[slice_list[2]].LazySlice[slice_list[3]].LazyTranspose(shuffle_list[2]).LazySlice[slice_list[4]].LazyTranspose()[:])

    os.remove('testfile.hdf5')

if __name__ == "__main__":
    main()
