import h5py
import numpy as np
from lazy_loading import DatasetView
import secrets
from random import shuffle


def main():
    f = h5py.File('testfile.hdf5')  # testfile.hdf5 shape is (10, 100000,30)
    dset = f['testdataset']
    dsetview = DatasetView(dset)

    secret_rand = secrets.SystemRandom()
    randslice = lambda d: slice(*secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(1, dsetview.shape[d]), 1))

    for _ in range(200):
        slice_list = [np.index_exp[:],  # randslice(0),np.s_[:],np.s_[:]
                      np.index_exp[:],
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape))),
                      tuple(randslice(i) for i in range(len(dsetview.shape)))]

        slice_0 = secrets.choice(slice_list)
        slice_1 = secrets.choice(slice_list)
        slice_2 = secrets.choice(slice_list)
        slice_3 = secrets.choice(slice_list)
        slice_4 = secrets.choice(slice_list)

        shuffle_a = [0, 1, 2]
        shuffle_b = [0, 1, 2]
        shuffle_c = [0, 1, 2]
        shuffle_d = [0, 1, 2]
        shuffle_e = [0, 1, 2]

        shuffle(shuffle_a)
        shuffle(shuffle_b)
        shuffle(shuffle_c)
        shuffle(shuffle_d)
        shuffle(shuffle_e)

        print("S_0", slice_0)
        print("S_1", slice_1)
        print("SSSSz_2", slice_2)
        print("SSSS_3", slice_3)
        print("SSSS_3", slice_4)

        print("shuffle_a", shuffle_a)
        print("shuffle_b", shuffle_b)
        print("shuffle_c", shuffle_c)
        print("shuffle_d", shuffle_d)
        print("shuffle_e", shuffle_e)

        assert (np.sum(
            np.abs(dsetview[slice_1][slice_2] - dsetview.LazySlice[slice_1].LazySlice[slice_2].dsetread()))) == 0

        assert (np.sum(np.abs(
            dsetview[slice_1][slice_2][slice_3] - dsetview.LazySlice[slice_1].LazySlice[slice_2].LazySlice[
                slice_3].dsetread()))) == 0
        assert (np.sum(np.abs(
            dset[slice_1][slice_2][slice_3] - dsetview.LazySlice[slice_1].LazySlice[slice_2].LazySlice[
                slice_3].dsetread()))) == 0

        assert (np.sum(
            np.abs(dset[slice_1].transpose() - dsetview.LazySlice[slice_1].lazy_transpose().dsetread()))) == 0

        assert (np.sum(
            np.abs(dsetview[slice_1].transpose() - dsetview.LazySlice[slice_1].lazy_transpose().dsetread()))) == 0
        assert (np.sum(np.abs(dsetview[slice_1].transpose()[slice_2] -
                              dsetview.LazySlice[slice_1].lazy_transpose().LazySlice[slice_2].dsetread()))) == 0
        assert (np.sum(np.abs(dsetview[slice_1].transpose()[slice_2][slice_3] -
                              dsetview.LazySlice[slice_1].lazy_transpose().LazySlice[slice_2].LazySlice[
                                  slice_3].dsetread()))) == 0

        assert (np.sum(np.abs(dset[slice_1].transpose()[slice_2][slice_3] -
                              dsetview.LazySlice[slice_1].lazy_transpose().LazySlice[slice_2].LazySlice[
                                  slice_3].dsetread()))) == 0

        assert (np.sum(np.abs(dset[slice_1].transpose([2, 0, 1]) -
                              dsetview.LazySlice[slice_1].lazy_transpose([2, 0, 1]).dsetread()))) == 0

        assert (np.sum(np.abs(dsetview[slice_1].transpose([2, 0, 1])[slice_2] -
                              dsetview.LazySlice[slice_1].lazy_transpose([2, 0, 1]).LazySlice[
                                  slice_2].dsetread()))) == 0

        assert (np.sum(np.abs(dsetview[slice_1].transpose([2, 0, 1])[slice_2][slice_3] -
                              dsetview.LazySlice[slice_1].lazy_transpose([2, 0, 1]).LazySlice[slice_2].LazySlice[
                                  slice_3].dsetread()))) == 0

        assert (np.sum(np.abs(dset[slice_1].transpose([2, 0, 1])[slice_2][slice_3].transpose([1, 2, 0]) -
                              dsetview.LazySlice[slice_1].lazy_transpose([2, 0, 1]).LazySlice[slice_2].LazySlice[
                                  slice_3].lazy_transpose([1, 2, 0]).dsetread()))) == 0

        assert (np.sum(np.abs(
            dset[:, :, :].transpose(shuffle_a)[slice_1].transpose(shuffle_b)[slice_2][slice_3].transpose(shuffle_c)[
                slice_4].transpose() -
            dsetview.lazy_transpose(shuffle_a).LazySlice[slice_1].transpose(shuffle_b).LazySlice[slice_2].LazySlice[
                slice_3].lazy_transpose(shuffle_c).LazySlice[slice_4].lazy_transpose().dsetread()))) == 0


if __name__ == "__main__":
    main()
