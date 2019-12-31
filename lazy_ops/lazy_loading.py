"""Provides a class to allow for lazy transposing and slicing operations on h5py datasets
Example Usage:
import h5py
from lazy_ops import DatasetView


dsetview = DatasetView(dataset) # dataset is an instantiated h5py dataset
view1 = dsetview.lazy_slice[1:10:2,:,0:50:5].lazy_transpose([2,0,1]).lazy_slice[25:55,1,1:4:1,:]
A = view1[:]          # Brackets on DataSetView call the h5py slicing method, that returns dataset data
B = view1.dsetread()  # same as view1[:]

"""

import h5py
import numpy as np
import sys

class DatasetView(h5py.Dataset):

    def __init__(self, dataset: h5py.Dataset = None, slice_index=(np.index_exp[:],()), axis_order=None):
        """
        Args:
          dataset:    the underlying dataset
          slice_index:  the aggregate slice and int indices after multiple lazy calls
          axis_order: the aggregate axis_order after multiple transpositions
        Returns:
          lazy object of the view
        """

        h5py.Dataset.__init__(self, dataset.id)
        if axis_order is None:
            self._axis_order = tuple(range(len(dataset.shape)))
        else:
            self._axis_order = axis_order
        self._lazy_slice_call = False
        self._dataset = dataset
        self._shape, self._key, self._int_index, self._axis_order = self._slice_shape(slice_index)

    @property
    def lazy_slice(self):
        """ Indicator for lazy_slice calls """
        self._lazy_slice_call = True
        return self

    @property
    def dataset(self):
        return self._dataset

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return self.len()

    def len(self):
        return self.shape[0]

    @property
    def key(self):
        """ The self.key slice is passed to the lazy instance """
        return self._key

    @property
    def axis_order(self):
        return self._axis_order

    def _slice_tuple(self, key):
        """  Allows single slice and int function calls
        Args:
          key: A slice object, or int
        Returns:
          The slice object tuple
        """
        if isinstance(key, (slice,int)):
            key = key,
        else:
            key = *key,

        return key

    def _slice_shape(self, slice_):
        """  For an slice returned by _slice_composition function, finds the shape
        Args:
          slice_: The slice object
        Returns:
          slice_shape: Shape of the slice object
          slice_key: An equivalent slice tuple with positive starts and stops
        """
        int_ind = slice_[1]
        slice_ = self._slice_tuple(slice_[0])
        # converting the slice to regular slices that only contain integers
        slice_regindices = [slice(*slice_[i].indices(self.dataset.shape[self.axis_order[i]])) if isinstance(slice_[i],slice)
                            else slice_[i]
                            for i in range(len(slice_))]
        
        slice_shape = ()
        int_index = ()
        axis_order = ()
        for i in range(len(slice_)):
            if isinstance(slice_[i],slice):
                slice_start, slice_stop, slice_step = slice_regindices[i].start, slice_regindices[i].stop, slice_regindices[i].step
                if slice_step < 1:
                    raise ValueError("Slice step parameter must be positive")
                if slice_stop < slice_start:
                    slice_start = slice_stop
                    slice_regindices[i] = slice(slice_start, slice_stop, slice_step)
                slice_shape += (1 + (slice_stop - slice_start -1 )//slice_step if slice_stop != slice_start else 0,)
                axis_order += (self.axis_order[i],)
            elif isinstance(slice_[i],int):
                int_index += ((i,slice_[i],self.axis_order[i]),)
            else:
                # slice_[i] is an iterator of integers
                slice_shape += (len(slice_[i]),)
                axis_order += (self.axis_order[i],)
        slice_regindices = tuple(el for el in slice_regindices if not isinstance(el,int))
        axis_order += tuple(self.axis_order[len(axis_order)+len(int_index)::])
        int_index += int_ind
        slice_shape += self.dataset.shape[len(slice_shape)+len(int_index)::]

        return slice_shape, slice_regindices, int_index, axis_order

    def __getitem__(self, new_slice):
        """  supports python's colon slicing syntax 
        Args:
          new_slice:  the new slice to compose with the lazy instance's self.key slice
        Returns:
          lazy object of the view
        """
        key_reinit = self._slice_composition(new_slice)
        if self._lazy_slice_call:
            self._lazy_slice_call = False
            return DatasetView(self.dataset, (key_reinit, self._int_index), self.axis_order)
        return DatasetView(self.dataset, (key_reinit, self._int_index), self.axis_order).dsetread()

    def lazy_iter(self, axis=0):
        """ lazy iterator over the first axis
            Modifications to the items are not stored
        """
        for i in range(self._shape[axis]):
            yield self.lazy_slice[(*np.index_exp[:]*axis,i)]

    def __call__(self, new_slice):
        """  allows lazy_slice function calls with slice objects as input"""
        return self.__getitem__(new_slice)

    def dsetread(self):
        """ Returns the data
        Returns:
          numpy array
        """
        # Note: Directly calling regionref with slices with a zero dimension does not
        # retain shape information of the other dimensions
        lazy_axis_order = self.axis_order
        lazy_key = self.key
        axis_order_slice_iter = self.axis_order
        for ind in self._int_index:
            lazy_axis_order = lazy_axis_order[:ind[0]] + (ind[2],) + lazy_axis_order[ind[0]:]
            lazy_key = lazy_key[:ind[0]] + (ind[1],) + lazy_key[ind[0]:]
        for ind in reversed(self._int_index):
            axis_order_slice_iter = tuple(i if i<ind[2] else i-1 for i in axis_order_slice_iter)

        reversed_axis_order = sorted(range(len(lazy_axis_order)), key=lambda i: lazy_axis_order[i])
        reversed_slice_key = tuple(lazy_key[i] for i in reversed_axis_order if i < len(lazy_key))

        return self.dataset[reversed_slice_key].transpose(axis_order_slice_iter)

    def _slice_composition(self, new_slice):
        """  composes a new_slice with the self.key slice
        Args:
          new_slice: The new slice
        Returns:
          merged slice object
        """
        new_slice = self._slice_tuple(new_slice)
        new_slice = self._ellipsis_slices(new_slice)
        slice_result = ()
        # Iterating over the new slicing tuple to change the merged dataset slice.
        for i in range(len(new_slice)):
            if isinstance(new_slice[i],slice):
                if i < len(self.key):
                    # converting new_slice slice to regular slices,
                    # newkey_start, newkey_stop, newkey_step only contains positive or zero integers
                    newkey_start, newkey_stop, newkey_step = new_slice[i].indices(self._shape[i])
                    if newkey_step < 1:
                        # regionref requires step>=1 for dataset data calls
                        raise ValueError("Slice step parameter must be positive")
                    if newkey_stop < newkey_start:
                        newkey_start = newkey_stop
                    if isinstance(self.key[i],slice):
                        slice_result += (slice(min(self.key[i].start + self.key[i].step * newkey_start, self.key[i].stop),
                                         min(self.key[i].start + self.key[i].step * newkey_stop, self.key[i].stop),
                                         newkey_step * self.key[i].step),)
                    else:
                        # self.key[i] is an iterator of integers
                        slice_result += (self.key[i][new_slice[i]],)
                else:
                    slice_result += (slice(*new_slice[i].indices(self.dataset.shape[self.axis_order[i]])),)
            elif isinstance(new_slice[i],int):
                if i < len(self.key):
                    if new_slice[i] >= self._shape[i] or new_slice[i] <= ~self._shape[i]:
                        raise IndexError("Index %d out of range, dim %d of size %d" % (new_slice[i],i,self._shape[i]))
                    if isinstance(self.key[i],slice):
                        int_index = self.key[i].start + self.key[i].step*(new_slice[i]%self._shape[i])
                        slice_result += (int_index,)
                    else:
                        # self.key[i] is an iterator of integers
                        slice_result += (self.key[i][new_slice[i]],)
                else:
                    slice_result += (new_slice[i],)
            else:
                try:
                    if any(not isinstance(el,int) for el in new_slice[i]):
                        raise ValueError("Indices must be integers")
                    if i < len(self.key):
                        if any(el >= self._shape[i] or el <= ~self._shape[i] for el in new_slice[i]):
                            raise IndexError("Index %s out of range, dim %d of size %d" % (str(new_slice[i]),i,self._shape[i]))
                        if isinstance(self.key[i],slice):
                            slice_result += (tuple(self.key[i].start + self.key[i].step*(ind%self._shape[i]) for ind in new_slice[i]),)
                        else:
                            # self.key[i] is an iterator of integers
                            slice_result += (tuple(self.key[i][ind] for ind in new_slice[i]),)
                    else:
                        slice_result += (new_slice[i],)
                except:
                    raise IndexError("Indices must be either integers, iterators of integers, or slice objects")
        slice_result += self.key[len(new_slice):]

        return slice_result

    @property
    def T(self):
        """ Same as lazy_transpose() """
        return self.lazy_transpose()

    def lazy_transpose(self, axis_order=None):
        """ Array lazy transposition, no axis_order reverses the order of dimensions
        Args:
          axis_order: permutation order for transpose
        Returns:
          lazy object of the view
        """

        if axis_order is None:
            axis_order = tuple(reversed(range(len(self.axis_order))))

        axis_order_reinit = tuple(self.axis_order[i] if i < len(self.axis_order) else i for i in axis_order)
        key_reinit = tuple(self.key[i] if i < len(self.key) else np.s_[:] for i in axis_order)
        key_reinit += tuple(self.key[i] for i in self.axis_order if i not in axis_order_reinit)
        axis_order_reinit += tuple(i for i in self.axis_order if i not in axis_order_reinit)

        return DatasetView(self.dataset, (key_reinit, self._int_index), axis_order_reinit)

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """ Using dataset.read_direct, reads data into an existing array
        Args:
          dest: C-contiguous as required by Dataset.read_direct
          source_sel: new selection slice
          dest_sel: output selection slice
        Returns:
          numpy array
        """
        if source_sel is None:
            new_key, new_int_index, new_axis_order = self.key, self._int_index, self.axis_order
        else:
            key_reinit = self._slice_composition(source_sel)
            _, new_key, new_int_index, new_axis_order = self._slice_shape(key_reinit)
        axis_order_slice_iter = new_axis_order
        for ind in new_int_index:
            new_axis_order = new_axis_order[:ind[0]] + (ind[2],) + new_axis_order[ind[0]:]
            new_key = new_key[:ind[0]] + (ind[1],) + new_key[ind[0]:]
        for ind in reversed(new_int_index):
            axis_order_slice_iter = tuple(i if i<ind[2] else i-1 for i in axis_order_slice_iter)

        reversed_axis_order = sorted(range(len(new_axis_order)), key=lambda i: new_axis_order[i])
        reversed_slice_key = tuple(new_key[i] for i in reversed_axis_order if i < len(new_key))
        #convert reversed_slice_key to numpy.s_[<args>] format, expected by dataset.read_direct
        if len(reversed_slice_key) == 1:
            reversed_slice_key = reversed_slice_key[0]

        reversed_dest_shape = tuple(dest.shape[i] for i in reversed_axis_order if i < len(dest.shape))
        reversed_dest = np.empty(shape=reversed_dest_shape, dtype=dest.dtype)

        if dest_sel is None:
            reversed_dest_sel = dest_sel
        else:
            reversed_dest_sel = tuple(dest_sel[i] for i in reversed_axis_order if i < len(dest_sel))
            #convert reversed_dest_sel to numpy.s_[<args>] format, expected by dataset.read_direct
            if len(reversed_slice_key) == 1:
                reversed_slice_key = reversed_slice_key[0]

        self.dataset.read_direct(reversed_dest, source_sel=reversed_slice_key, dest_sel=reversed_dest_sel)
        np.copyto(dest, reversed_dest.transpose(axis_order_slice_iter))

    def _ellipsis_slices(self, new_slice):
        """ Change Ellipsis dimensions to slices
        Args:
          new_slice: The new slice
        Returns:
          equivalent slices with Ellipsis expanded
        """
        ellipsis_count = new_slice.count(Ellipsis)
        if ellipsis_count == 1:
            ellipsis_index = new_slice.index(Ellipsis)
            if ellipsis_index == len(new_slice)-1:
                new_slice = new_slice[:-1]
            else:
                num_ellipsis_dims = len(self.shape) - (len(new_slice) - 1)
                new_slice = new_slice[:ellipsis_index] + np.index_exp[:]*num_ellipsis_dims + new_slice[ellipsis_index+1:]
        elif ellipsis_count > 0:
            raise IndexError("Only a single Ellipsis is allowed")
        return new_slice

def lazy_transpose(dset: h5py.Dataset, axes=None):
    """ Array lazy transposition, not passing axis argument reverses the order of dimensions
    Args:
      dset: h5py dataset
      axes: permutation order for transpose
    Returns:
      lazy transposed DatasetView object
    """
    if axes is None:
        axes = tuple(reversed(range(len(dset.shape))))
    
    return DatasetView(dset).lazy_transpose(axis_order=axes)
