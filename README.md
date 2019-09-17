## lazy_ops

<strong>Provides a class to allow for lazy transposing and slicing operations on h5py datasets </strong>

Example Usage:

```python
import h5py
from lazy_loading import DatasetView

dsetview = DatasetView(dataset) # dataset is an instantiated h5py dataset
view1 = dsetview.LazySlice[1:10:2,:,0:50:5].lazy_transpose([2,0,1]).LazySlice[25:55,1,1:4:1,:].transpose()

A = view1.dsetread #returns the data for view1 of h5py dataset
B = dsetview[:] #Brackets without LazySlice call the h5py slicing method, that returns the data
```
