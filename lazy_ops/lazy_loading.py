"""Provides a class to allow for lazy transposing and slicing operations on h5py datasets
Example Usage:
import h5py
from lazy_ops import DatasetView


dsetview = DatasetView(dataset) # dataset is an instantiated h5py dataset
view1 = dsetview.LazySlice[1:10:2,:,0:50:5].LazyTranspose([2,0,1]).LazySlice[25:55,1,1:4:1,:]
A = view1[:]          # Brackets on DataSetView call the h5py slicing method, that returns dataset data
B = view1.dsetread()  # same as view1[:]

"""

import h5py
import numpy as np

class DatasetView(h5py.Dataset):

    def __init__(self, dataset : h5py.Dataset = None, slice_index=np.index_exp[:], axis_order=None):
        """
        Args:
          dataset:    the underlying dataset
          slice_index:  the aggregate slice after multiple lazy slicing
          axis_order: the aggregate axis_order after multiple transpositions
        Returns:
          lazy object of the view
        """

        h5py.Dataset.__init__(self, dataset.id)
        if axis_order is None:
            self._axis_order = list(range(len(dataset.shape)))
        else:
            self._axis_order = axis_order
        self._lazy_slice_call = False
        self._dataset = dataset
        self._LazyShape, self._key = self._slice_shape(slice_index)
        self._LazySlice = self

    @property
    def LazySlice(self):
        ''' Indicator for LazySlice calls '''
        self._lazy_slice_call = True
        return self._LazySlice

    @property
    def dataset(self):
        return self._dataset

    @property
    def LazyShape(self):
        """ The shape due to the lazy operations  """
        return self._LazyShape

    @property
    def key(self):
        """ The self.key slice is passed to the lazy instance and is not altered by the instance's init call """
        return self._key

    @property
    def axis_order(self):
        return self._axis_order

    def _slice_tuple(self, key):
        """  Allows single slice function calls
        Args:
          key: The slice object
        Returns:
          The slice object tuple
        """
        if isinstance(key, slice):
            key = key,
        else:
            key = *key,

        return key

    def _slice_shape(self,slice_):
        """  For an slice returned by slice_composition function, finds the shape
        Args:
          slice_: The slice object
        Returns:
          slice_shape: Shape of the slice object
          slice_key: An equivalent slice tuple with positive starts and stops
        """
        slice_ = self._slice_tuple(slice_)
        # converting the slice to regular slices that only contain integers
        slice_regindices = [slice(*slice_[i].indices(self.dataset.shape[self.axis_order[i]])) for i in range(len(slice_))]
        slice_shape = ()
        for i in range(len(slice_)):
            slice_start, slice_stop, slice_step = slice_regindices[i].start, slice_regindices[i].stop, slice_regindices[i].step
            if slice_step < 1:
                raise ValueError("Slice step parameter must be positive")
            if slice_stop < slice_start:
                slice_start = slice_stop
                slice_regindices[i] = slice(slice_start, slice_stop, slice_step)
            slice_shape += (1 + (slice_stop - slice_start -1 )//slice_step if slice_stop != slice_start else 0,)

        slice_regindices = tuple(slice_regindices)
        return slice_shape, slice_regindices

    def __getitem__(self, new_slice):
        """  supports python's colon slicing syntax 
        Args:
          new_slice:  the new slice to compose with the lazy instance's self.key slice
        Returns:
          lazy object of the view
        """
        if self._lazy_slice_call:
            self._lazy_slice_call = False
            new_slice = self._slice_tuple(new_slice)
            self.key_reinit = self.slice_composition(new_slice)
            return DatasetView(self.dataset, self.key_reinit, self.axis_order)

        return self.dsetread()[new_slice]

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
        self.reversed_axis_order = sorted(range(len(self.axis_order)), key=lambda i: self.axis_order[i])
        reversed_slice_key = tuple(self.key[i] for i in self.reversed_axis_order if i < len(self.key))
        return self.dataset[reversed_slice_key].transpose(self.axis_order)

    def slice_composition(self, new_slice):
        """  composes a new_slice with the self.key slice
        Args:
          new_slice: The new slice
        Returns:
          merged slice object
        """
        slice_result = ()
        # Iterating over the new slicing tuple to change the merged dataset slice.
        for i in range(len(new_slice)):
            if i < len(self.key):
                # converting new_slice slice to regular slices,
                # newkey_start, newkey_stop, newkey_step only contains positive or zero integers
                newkey_start, newkey_stop, newkey_step = new_slice[i].indices(self.LazyShape[i])
                if newkey_step < 1:
                    # regionref requires step>=1 for dataset data calls
                    raise ValueError("Slice step parameter must be positive")
                if newkey_stop < newkey_start:
                    newkey_start = newkey_stop

                slice_result += (slice(min(self.key[i].start + self.key[i].step * newkey_start, self.key[i].stop), 
                                 min(self.key[i].start + self.key[i].step * newkey_stop , self.key[i].stop),
                                 newkey_step * self.key[i].step),)
            else:
                slice_result += (slice(*new_slice[i].indices(self.dataset.shape[self.axis_order[i]])),)
        for i in range(len(new_slice), len(self.key)):
            slice_result += (slice(*self.key[i].indices(self.dataset.shape[self.axis_order[i]])),)

        return slice_result

    def transpose(self, axis_order=None):
        """ Returns the transposed data
        Args:
          axis_order: permutation order for data transpose
        Returns:
          numpy array
        """
        return self.dsetread().transpose(axis_order)

    def T(self, axis_order=None):
        """ Same as transpose() """
        return self.transpose(axis_order)

    def LazyTranspose(self, axis_order=None):
        """ Array lazy transposition, no axis_order reverses the order of dimensions
        Args:
          axis_order: permutation order for transpose
        Returns:
          lazy object of the view
        """

        if axis_order is None:
            axis_order = list(reversed(range(len(self.axis_order))))

        self.key_reinit = [self.key[i] if i < len(self.key) else np.s_[:] for i in axis_order]

        axis_order_reinit = [self.axis_order[i] if i < len(self.axis_order) else i for i in axis_order]
        return DatasetView(self.dataset, self.key_reinit, axis_order_reinit)

