# lazy_ops
[![PyPI version](https://badge.fury.io/py/nwbwidgets.svg)](https://badge.fury.io/py/nwbwidgets)
[![codecov](https://codecov.io/gh/catalystneuro/lazy_ops/branch/master/graph/badge.svg)](https://codecov.io/gh/catalystneuro/lazy_ops)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<strong>Lazy transposing and slicing of h5py Datasets and zarr arrays</strong>

## Installation

```bash
$ pip install lazy_ops
```

## Usage:

```python
from lazy_ops import DatasetView

# h5py #
import h5py
dsetview = DatasetView(dataset) # dataset is an instance of h5py.Dataset
view1 = dsetview.lazy_slice[1:40:2,:,0:50:5].lazy_transpose([2,0,1]).lazy_slice[8,5:10]

# zarr #
import zarr
zarrview = DatasetView(zarray) # dataset is an instance of zarr.core.Array
view1 = zview.lazy_slice[1:10:2,:,5:10].lazy_transpose([0,2,1]).lazy_slice[0:3,1:4]

# reading from view on either h5py or zarr
A = view1[:]          # Brackets on DataSetView call the h5py or zarr slicing method, returning the data
B = view1.dsetread()  # same as view1[:]

# iterating on either h5yy or zarr
for ib in view.lazy_iter(axis=1):
    print(ib[0])

```
