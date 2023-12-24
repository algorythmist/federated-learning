import numpy as np

from simplefql.util import shuffle_together


def test_shuffle_together():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6, 8, 10])
    c = np.array([3, 6, 9, 12, 15])
    a, b, c = shuffle_together(a, b, c)
    for i in range(len(a)):
        assert a[i] == b[i] // 2
        assert a[i] == c[i] // 3
