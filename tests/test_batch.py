import os
import h5py
import numpy as np
from lazy_ops import DatasetView, lazy_transpose
import secrets
from random import shuffle
from numpy.testing import assert_array_equal
import unittest
import itertools

def main():

    if os.path.isfile('testfile.hdf5'):
        os.remove('testfile.hdf5')
    f = h5py.File('testfile.hdf5','x')
    ndims = 4
    secret_rand = secrets.SystemRandom()
    dset = f.create_dataset(name='test_dataset', data=np.random.rand(*secret_rand.sample(range(1, 200//ndims), ndims)))
    dsetview = DatasetView(dset)

    randslice = lambda d: slice(*secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(-dsetview.shape[d] * 5 // 4, dsetview.shape[d] * 5 // 4), 1),
                                *secret_rand.sample(range(1, dsetview.shape[d] + 1), 1))

    for _ in range(50):

        slice_list = [randslice(0),
                      np.s_[:],
                      np.index_exp[:],
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))),
                      tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1)))]

        for c in range(5):
            slice_list += [(tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape))))+
                           (Ellipsis,)+
                           tuple(randslice(i) for i in range(secrets.randbelow(len(dsetview.shape)+1))))[:len(dsetview.shape)]]

        for c in range(5):
            slice_list_d = ()
            for d in range(secret_rand.randrange(len(dsetview.shape))):
                slice_list_d += (secret_rand.choice([randslice(d),
                                 secret_rand.randrange(dsetview.shape[d]),
                                 tuple(sorted(set(secret_rand.randrange(dsetview.shape[d]) for i in range(secret_rand.randrange(min(10,dsetview.shape[d]))))))
                                                    ]),)
            slice_list_d += (Ellipsis,)*secret_rand.randrange(2)
            for d in range(secret_rand.randrange(len(slice_list_d), len(dsetview.shape)+1),
                           len(dsetview.shape)):
                slice_list_d += (secret_rand.choice([randslice(d),
                                  secret_rand.randrange(dsetview.shape[d]),
                                  tuple(sorted(set(secret_rand.randrange(dsetview.shape[d]) for i in range(secret_rand.randrange(min(10,dsetview.shape[d]))))))
                                                    ]),)
            slice_list.append(slice_list_d)
                        
        num_slices = 15
        slice_list = [secrets.choice(slice_list) for _ in range(num_slices)]

        num_list = 10
        shuffle_list = [list(range(ndims)) for _ in range(num_list)]
        for li in shuffle_list:
            shuffle(li)

        # test lazy_transpose
        assert_array_equal(np.transpose(dset[:],axes=shuffle_list[0]), lazy_transpose(dset, axes=shuffle_list[0]))

        for _ in range(80):
            string_dset = 'dset'
            string_dsetview = 'dsetview'
            secret_b = secrets.SystemRandom()
            num_ops = secret_b.randrange(15)
            for op_i in range(num_ops):
                num_list_rand = secret_b.randrange(num_list)
                num_slice_rand = secret_b.randrange(num_slices)
                op_index_rand = secret_b.randrange(4)
                shuffle_list_el = shuffle_list[num_list_rand]
                slice_list_el = slice_list[num_slice_rand]
                op_i_string_dset = ('.transpose('+str(shuffle_list_el)+')',
                                    '.transpose()',
                                    '.T',
                                    '['+str(slice_list_el)+']')
                op_i_string_dsetview = ('.lazy_transpose('+str(shuffle_list_el)+')',
                                        '.lazy_transpose()',
                                        '.T',
                                        '.lazy_slice['+str(slice_list_el)+']')
                string_dset = string_dset + op_i_string_dset[op_index_rand]
                string_dsetview = string_dsetview + op_i_string_dsetview[op_index_rand]
            if string_dset[0:5] != 'dset.': # drop starting with transpose
                valid = False
                valid_iter = False
                try:
                    eval_string_dset = eval(string_dset)
                    indexing_vector_count = 0
                    indexing_int_count = 0
                    indexing_slice_count = 0
                    slice_at_index = np.Inf
                    array_at_index = -1
                    eval_string_dsetview = eval(string_dsetview)
                    str_key = eval_string_dsetview._key
                    for k in range(len(str_key)):
                        if isinstance(str_key[k],int):
                            indexing_int_count += 1
                        elif isinstance(str_key[k],slice):
                            if indexing_slice_count < k:
                                slice_at_index = min(k,slice_at_index)
                            indexing_slice_count += 1
                        else:
                            indexing_vector_count += 1
                            array_at_index = max(k,array_at_index)
                    if indexing_vector_count>1:
                        raise Exception("h5py does not support more than one array indexing")
                    if len(eval_string_dset.shape)>1:
                        if slice_at_index < array_at_index:
                            raise Exception("numpy Fancy indexing transposes and does not give the expected results, "+
                                            "and it does not match with h5py's indexing, which are as expected. Out of scope")
                    valid = True
                    valid_iter = True
                except:
                    pass
                if valid and np.prod(eval_string_dset.shape)!=0:
                    try:
                        assert_array_equal(eval_string_dset,eval(string_dsetview+'[()]'))
                    except Exception as e:
                        print(slice_at_index,array_at_index)
                        print(str_key)
                        print(shuffle_list)
                        print(slice_list)
                        print(dset.shape)
                        print(dsetview.shape)
                        print(string_dset)
                        print(string_dsetview)
                        raise e
                    if valid_iter:
                        try:
                            i = 0
                            for dsetview_lazy_i in eval_string_dsetview.lazy_iter():
                                assert_array_equal(eval_string_dset[i],dsetview_lazy_i)
                                i += 1
                            for axis in range(len(eval_string_dsetview.shape)):
                                i = 0
                                for dsetview_lazy_i in eval_string_dsetview.lazy_iter(axis = axis):
                                    assert_array_equal(eval_string_dset[(*np.index_exp[:]*axis,i)],dsetview_lazy_i)
                                    i += 1
                        except Exception as e:
                            print(slice_at_index,array_at_index)
                            print(str_key)
                            print(shuffle_list)
                            print(slice_list)
                            print(dset.shape)
                            print(dsetview.shape)
                            print(string_dset)
                            print(string_dsetview)
                            raise e

    os.remove('testfile.hdf5')

if __name__ == "__main__":
    main()
