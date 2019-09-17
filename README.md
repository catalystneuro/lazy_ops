<strong>Provides a class to allow for lazy transposing and slicing operations on h5py datasets </strong>

Example Usage:

1\)

import h5py

from lazy_loading import DataSetViews

dsetviews = DataSetViews(dataset) # dataset is an instantiated h5py dataset


view1 = dsetviews.lazy_slice[1:10:2,:,0:50:5].lazy_transpose([2,0,1]).lazy_slice[25:55,1,1:4:1,:].transpose()

A = view1.dsetread #returns the data for view1 of h5py dataset

B = dsetviews[:] #Brackets without lazy_slice call the h5py slicing method, that returns the data
