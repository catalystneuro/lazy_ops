import os
import h5py
import numpy as np
from lazy_ops import DatasetView
import secrets
from random import shuffle
from numpy.testing import assert_array_equal


def test_func():

    if os.path.isfile('testfile.hdf5'):
        os.remove('testfile.hdf5')
    f = h5py.File('testfile.hdf5')
    ndims = 4
    secret_rand = secrets.SystemRandom()
    dset = f.create_dataset(name='test_dataset', data=np.random.rand(*secret_rand.sample(range(1, 200//ndims), ndims)))
    dsetview = DatasetView(dset)

    randslice = lambda d: slice(*secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(1, dsetview.shape[d] + 1), 1))

    for _ in range(5):

        slice_list = [randslice(0),
                      np.s_[:],
                      np.index_exp[:],
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1)))]

        slice_list = [secrets.choice(slice_list) for _ in range(5)]

        shuffle_list = [list(range(ndims)) for _ in range(5)]
        for li in shuffle_list:
            shuffle(li)

        assert_array_equal(dset[slice_list[1]],
                           dsetview.lazy_slice[slice_list[1]][:])

        assert_array_equal(dsetview[slice_list[1]][slice_list[2]],
                           dsetview.lazy_slice[slice_list[1]].lazy_slice[slice_list[2]][:])

        assert_array_equal(dsetview[slice_list[1]][slice_list[2]][slice_list[3]],
                           dsetview.lazy_slice[slice_list[1]].lazy_slice[slice_list[2]].lazy_slice[slice_list[3]][:])

        assert_array_equal(dset[slice_list[1]][slice_list[2]][slice_list[3]],
                           dsetview.lazy_slice[slice_list[1]].lazy_slice[slice_list[2]].lazy_slice[slice_list[3]][:])

        assert_array_equal(dset[slice_list[1]][:].transpose(), dsetview.lazy_slice[slice_list[1]].lazy_transpose()[:])

        assert_array_equal(dsetview[slice_list[1]][:].transpose(), dsetview.lazy_slice[slice_list[1]].lazy_transpose()[:])

        assert_array_equal(dsetview[slice_list[1]][:].transpose()[slice_list[2]],
                           dsetview.lazy_slice[slice_list[1]].lazy_transpose().lazy_slice[slice_list[2]][:])

        assert_array_equal(dsetview[slice_list[1]][:].transpose()[slice_list[2]][slice_list[3]],
                           dsetview.lazy_slice[slice_list[1]].lazy_transpose().lazy_slice[slice_list[2]].
                           lazy_slice[slice_list[3]][:])

        assert_array_equal(dset[slice_list[1]][:].transpose()[slice_list[2]][slice_list[3]],
                           dsetview.lazy_slice[slice_list[1]].lazy_transpose().lazy_slice[slice_list[2]].
                           lazy_slice[slice_list[3]][:])

        assert_array_equal(dsetview[slice_list[1]][:].transpose(shuffle_list[3])[slice_list[2]][slice_list[3]],
                           dsetview.lazy_slice[slice_list[1]].lazy_transpose(shuffle_list[3]).lazy_slice[slice_list[2]].
                           lazy_slice[slice_list[3]].dsetread())

        assert_array_equal(dset[slice_list[1]][:].transpose(shuffle_list[3])[slice_list[2]][slice_list[3]].
                           transpose(shuffle_list[4]),
                           dsetview.lazy_slice[slice_list[1]].lazy_transpose(shuffle_list[3]).lazy_slice[slice_list[2]]
                           .lazy_slice[slice_list[3]].lazy_transpose(shuffle_list[4])[:])

        assert_array_equal(dset[:][:].transpose(shuffle_list[0]), dsetview.lazy_transpose(shuffle_list[0]).dsetread()[:])

        assert_array_equal(dsetview[:].transpose(shuffle_list[0])[slice_list[1]][:].transpose(shuffle_list[1])[slice_list[2]]
                           [slice_list[3]][:].transpose(shuffle_list[2])[slice_list[4]][:].transpose(),
                           dsetview.lazy_transpose(shuffle_list[0]).lazy_slice[slice_list[1]].
                           lazy_transpose(shuffle_list[1]).lazy_slice[slice_list[2]].lazy_slice[slice_list[3]].
                           lazy_transpose(shuffle_list[2]).lazy_slice[slice_list[4]].lazy_transpose()[:])

        assert_array_equal(dset[:].transpose(shuffle_list[0])[slice_list[1]][:].transpose(shuffle_list[1])[slice_list[2]]
                           [slice_list[3]][:].
                           transpose(shuffle_list[2])[slice_list[4]][:].transpose(),
                           dsetview.lazy_transpose(shuffle_list[0]).lazy_slice[slice_list[1]].
                           lazy_transpose(shuffle_list[1]).lazy_slice[slice_list[2]].lazy_slice[slice_list[3]].
                           lazy_transpose(shuffle_list[2]).lazy_slice[slice_list[4]].lazy_transpose()[:])

    os.remove('testfile.hdf5')



