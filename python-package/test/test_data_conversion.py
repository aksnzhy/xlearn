# Copyright (c) 2018 by contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
# This file test the data conversion for sklearn API
import unittest
import tempfile

import numpy as np
from xlearn import write_data_to_xlearn_format
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file

class TestDataConversion(unittest.TestCase):
    """
    Test data conversion to libsvm and libffm inside LRModel, FMModel and FFMModel
    """

    def setUp(self):
        # data generation
        self.num_rows = 10
        self.num_features = 4
        self.X = np.random.randn(self.num_rows, self.num_features)
        self.X[self.X < 0] = 0 # introduce sparsity
        self.y = np.random.binomial(1, 0.5, size=(self.num_rows, 1))
        self.fields = np.array([1, 2, 1, 0])

    def _read_libffm_file(self, filename):
        """
        An internal function for reading libffm back to numpy array.
        """

        X_true = np.zeros((self.num_rows, self.num_features))
        y_true = np.zeros((self.num_rows, 1))
        field_true = np.zeros((self.num_features, 1))
        with open(filename, 'r') as f:
            i = 0
            for line in f:
                tmp_row = line.replace('\n', '').split(' ')

                # extract label
                y_true[i] = int(tmp_row[0])

                # extract data and fields
                for k in range(1, len(tmp_row)):
                    if len(tmp_row[k]) > 0:
                        tmp_str = tmp_row[k].split(':')
                        j = int(tmp_str[1])
                        field_true[j] = int(tmp_str[0])
                        tmp_data = float(tmp_str[2])
                        X_true[i, j] = tmp_data
                i = i + 1

        return X_true, y_true, field_true

    def test_convert_numpy_to_libsvm(self):
        """
        Test if the conversion between libsvm and numpy array is correct
        """

        file = tempfile.NamedTemporaryFile(delete=True)

        # write to temporary files
        write_data_to_xlearn_format(self.X, self.y, file.name)

        # load data back and compare if they are the same as original data
        X_true, y_true = load_svmlight_file(file.name)
        file.close()

        assert np.all(np.isclose(self.X, X_true.todense()))
        assert np.all(self.y.ravel() == y_true.ravel())

    def test_convert_csr_to_libsvm(self):
        """
        Test if the conversion between libsvm and csr matrix is correct
        """
        X_spase = csr_matrix(self.X)
        file = tempfile.NamedTemporaryFile(delete=True)

        # write to temporary files
        write_data_to_xlearn_format(X_spase, self.y, file.name)

        # load data back and compare if they are the same as original data
        X_true, y_true = load_svmlight_file(file.name)
        file.close()

        assert np.all(np.isclose(X_spase.todense(), X_true.todense()))
        assert np.all(self.y.ravel() == y_true.ravel())

    def test_convert_numpy_to_libffm(self):
        """
        Test if the conversion between libffm and numpy array is correct
        """
        file = tempfile.NamedTemporaryFile(delete=True)

        # write data to libffm format
        write_data_to_xlearn_format(self.X, self.y, file.name, fields=self.fields)

        # read back data from file
        X_true, y_true, field_true = self._read_libffm_file(file.name)
        file.close()

        assert np.all(np.isclose(self.X, X_true))
        assert np.all(self.y.ravel() == y_true.ravel())
        assert np.all(self.fields.ravel() == field_true.ravel())

    def test_convert_csr_to_libffm(self):
        """
        Test if the conversion between libffm and csr matrix is correct
        """
        X_sparse = csr_matrix(self.X)
        file = tempfile.NamedTemporaryFile(delete=True)

        # write data to libffm format
        write_data_to_xlearn_format(X_sparse, self.y, file.name, fields=self.fields)

        # read back data from file
        X_true, y_true, field_true = self._read_libffm_file(file.name)
        file.close()

        assert np.all(np.isclose(X_sparse.todense(), X_true))
        assert np.all(self.y.ravel() == y_true.ravel())
        assert np.all(self.fields.ravel() == field_true.ravel())

if __name__ == '__main__':
    unittest.main()