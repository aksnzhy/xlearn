#!/usr/bin/python
# coding: utf-8

import numpy as np
import xlearn as xl

from scipy.sparse import csr_matrix

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr = csr_matrix((data, (row, col)), shape=(3, 3))

xl.DMatrix(csr, field=csr)


