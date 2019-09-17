"""Provides a class to allow for lazy transposing and slicing operations on h5py datasets

Example Usage:

import h5py
from lazy_loading import DataSetView
.
.
dsetview = DatasetView(dataset) # dataset is an instantiated h5py dataset
view1 = dsetview.lazy_slice[1:10:2,:,0:50:5].lazy_transpose([2,0,1]).lazy_slice[25:55,1,1:4:1,:].transpose()

A = view1.dsetread  # reads view1 of h5py dataset data
B = dsetview[:] # Brackets on DataSetView without lazy_slice call the h5py slicing method, that returns dataset data
"""

import h5py
import numpy as np


class DatasetView(h5py.Dataset):
    """ Inherit from this class when implementing dataset views """

    def __init__(self, dataset: h5py.Dataset):
        """
        Args:
          dataset: Underlying HDF5 dataset returned from h5py
        Returns:
          Dataset view object
        """
        self.__slice_key = np.index_exp[:]
        h5py.Dataset.__init__(self, dataset.id)
        self.lazy_slice = LazySlice(self)

    def dsetread(self):
        """ Returns the data
        Returns:
          numpy array
        """
        return self[:]

    def lazy_transpose(self, axis_order=None):
        """ Array lazy transposition
        Args:
          axis_order: permutation order for transpose
        Returns:
          dataset view object
        """
        return self.lazy_slice.lazy_transpose(axis_order)


class LazySlice(object):
    def __init__(self, dview, key=np.index_exp[:], axis_order=None):
        """
        Args:
          dview:      the underlying view object referring to the dataset
          key:        the aggregate slice after multiple lazy slicing
          axis_order: the aggregate axis_order after multiple transpositions
        Returns:
          lazy object of the view
        """
        if axis_order is None:
            self._axis_order = list(range(len(dview.shape)))
        else:
            self._axis_order = axis_order
        self._key = key
        self._dview = dview
        self.lazy_slice = self

    @property
    def dview(self):
        return self._dview

    @property
    def key(self):
        """ the self.key slice is passed to the lazy instance and is not altered the instance's init call """
        return self._key

    @property
    def axis_order(self):
        return self._axis_order

    def __getitem__(self, new_slice):
        """  supports python's colon slicing syntax """
        """
        Args:
          new_slice:  the new slice to compose with the lazy instance's self.key slice
        Returns:
          lazy object of the view
        """
        if isinstance(new_slice, slice):
            new_slice = new_slice,
        else:
            new_slice = *new_slice,
        self.key_reinit = self.slice_composition(new_slice)
        return LazySlice(self.dview, self.key_reinit, self.axis_order)

    def __call__(self, new_slice):
        """  allows LazySlice function calls with slice objects as input"""
        return self.__getitem__(new_slice)

    def dsetread(self):
        """ Returns the data
        Returns:
          numpy array
        """
        # Note: Directly calling regionref with slices with a zero dimension does not
        # retain shape information of the other dimensions
        self.reversed_axis_order = sorted(range(len(self.key)), key=lambda i: self.axis_order[i])
        reversed_slice_key = tuple(self.key[i] for i in self.reversed_axis_order)
        return self.dview[reversed_slice_key].transpose(self.axis_order)

    def slice_composition(self, new_slice):
        """  composes a new_slice with the self.key slice
        Args:
          new_slice: The new slice
        Returns:
          merged slice object
        """
        slice_list = []
        # Iterating over the new slicing tuple to change the merge dataset slice.
        for i in range(len(new_slice)):
            if i < len(self.key):
                # converting last stored key slice to regular slices that only contain integers
                pre_start, pre_stop, pre_step = self.key[i].indices(self.dview.shape[self.axis_order[i]])
                assert pre_start >= 0 and pre_stop >= 0 and pre_step >= 1
                if pre_stop < pre_start:
                    pre_start = pre_stop
                pre_shape = 1 + (pre_stop - pre_start -1 )//abs(pre_step) if pre_stop != pre_start else 0  # array dimension after last slice

                # converting new_slice slice to regular slices that only contain integers
                newkey_directslice = new_slice[i].indices(pre_shape)
                newkey_start, newkey_stop, newkey_step = newkey_directslice
                # newkey_directslice only contains positive or zero integers
                # regionref requires step>=1 for dataset data calls
                assert newkey_start >= 0 and newkey_stop >= 0 and newkey_step >=1
                slice_list.append(slice(pre_start + pre_step * newkey_start, min(pre_start + pre_step * newkey_stop , pre_stop), newkey_step * pre_step))
            else:
                slice_list.append(slice(*new_slice[i].indices(self.dview.shape[self.axis_order[i]])))
        for i in range(len(new_slice), len(self.key)):
            slice_list.append(slice(*self.key[i].indices(self.dview.shape[self.axis_order[i]])))
        slice_result = tuple(slice_list)
        return slice_result

    def transpose(self, axis_order=None):
        """ Same as lazy_transpose() """
        return self.lazy_transpose(axis_order)

    def T(self, axis_order=None):
        """ Same as lazy_transpose() """
        return self.lazy_transpose(axis_order)

    def lazy_transpose(self, axis_order=None):
        """ Array lazy transposition, no axis_order reverses the order of dimensions
        Args:
          axis_order: permutation order for transpose
        Returns:
          lazy ojbect of the view
        """

        if axis_order is None:
            axis_order = list(reversed(range(len(self.axis_order))))

        self.key_reinit = [self.key[i] if i < len(self.key) else np.s_[:] for i in axis_order]

        axis_order_reinit = [self.axis_order[i] for i in axis_order]
        return LazySlice(self.dview, self.key_reinit, axis_order_reinit)


