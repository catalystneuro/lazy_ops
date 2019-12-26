# lazy_ops

<strong>Provides a class to allow for lazy transposing, slicing, and iterating operations on h5py datasets </strong>

## Installation

```bash
$ pip install lazy_ops
```

## Usage:

```python
import h5py
from lazy_ops import DatasetView

dsetview = DatasetView(dataset) # dataset is an instantiated h5py dataset
view1 = dsetview.lazy_slice[1:10:2,:,0:50:5].lazy_transpose([2,0,1]).lazy_slice[25:55,1,1:4:1,:].lazy_transpose()

A = view1[:]          # Brackets on DataSetView call the h5py slicing method, that returns the data
B = view1.dsetread()  # same as view1[:]

for ib in view1.lazy_iter(axis=1):
    print(ib[0])

```



