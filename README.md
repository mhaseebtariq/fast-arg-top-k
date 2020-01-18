### Get the indices of the top K values in an (1-D) array
The implementation uses a function compiled using Numba; and is more than <b>50x</b> faster than using numpy.argsort(...)!


```python
import sys

try:
    import numpy as np
except ImportError:
    !{sys.executable} -m pip install numpy==1.17.4
try:
    import numba as nb
except ImportError:
    !{sys.executable} -m pip install numba==0.45.1

import numpy as np
import numba as nb
```


```python
FLOAT_TYPE = np.float32
FLOAT_BUFFER = np.finfo(FLOAT_TYPE).resolution

K = 100
```


```python
@nb.njit(nb.types.Array(nb.int64, 1, "A")(nb.float32[:]))
def fast_arg_top_k(array):
    """
    Gets the indices of the top k values in an (1-D) array.
    * NOTE: The returned indexes are not sorted based on the top values
    """
    sorted_indexes = np.zeros((K,), dtype=FLOAT_TYPE)
    minimum_index = 0
    minimum_index_value = 0
    for value in array:
        if value > minimum_index_value:
            sorted_indexes[minimum_index] = value
            minimum_index = sorted_indexes.argmin()
            minimum_index_value = sorted_indexes[minimum_index]
    # FLOAT_BUFFER = np.finfo(FLOAT_TYPE).resolution
    # In some situations, because of different resolution you get k-1 results - this is to avoid that!
    minimum_index_value -= FLOAT_BUFFER
    return (array >= minimum_index_value).nonzero()[0][::-1][:K]
```


```python
def numpy_arg_top_k(array):
    return (-array).argsort()[:K]
```


```python
array = np.array(np.random.sample((1000000,)), dtype=FLOAT_TYPE)
```


```python
time_fast = %timeit -n 100 -o fast_arg_top_k(array)
```

    2.02 ms ± 37 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
time_numpy = %timeit -n 10 -o numpy_arg_top_k(array)
```

    104 ms ± 3.99 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
result_fast = sorted(fast_arg_top_k(array))
result_numpy = sorted(numpy_arg_top_k(array))

number_of_common = len(set(result_fast).intersection(result_numpy))
percentage_of_common = round((number_of_common / K) * 100)

# Could happen that there are a few exact same values in the top K
# In that case there could be a few differences
print(f'{percentage_of_common}% of the indices are same!')
```

    100% of the indices are same!



```python
print(f'"fast_arg_top_k" is around {round(time_numpy.best / time_fast.best)}x faster than "numpy_arg_top_k"!')
```

    "fast_arg_top_k" is around 50x faster than "numpy_arg_top_k"!
