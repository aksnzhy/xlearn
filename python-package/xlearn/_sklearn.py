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
import os
import shutil
import tempfile
import warnings
import numpy as np

from .xlearn import create_linear, create_fm, create_ffm
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y

def write_data_to_xlearn_format(X, y, filepath, fields=None):
    """ Write data to xlearn format (libsvm or libffm). Modified from
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/datasets/svmlight_format.py

    :param X: array-like
              Feature matrix in numpy or sparse format
    :param y: array-like
              Label in numpy or sparse format
    :param filepath: file location for writing data to
    :param fields: An array specifying fields in each columns of X. It should have same length 
        as the number of columns in X. When set to None, convert data to libsvm format else
        libffm format.
    """

    with open(filepath, "wb") as f_handle:
        X_is_sp = int(hasattr(X, "tocsr"))
        y_is_sp = int(hasattr(y, "tocsr"))

        if X.dtype.kind == 'i':
            value_pattern = u"%d:%d"
        else:
            value_pattern = u"%d:%.16g"

        if fields is not None:
            is_ffm_format = True
            value_pattern = u"%d:" + value_pattern
        else:
            is_ffm_format = False

        if y.dtype.kind == 'i':
            label_pattern = u"%d"
        else:
            label_pattern = u"%.16g"

        line_pattern = u"%s %s\n"

        for i in range(X.shape[0]):
            if X_is_sp:
                span = slice(X.indptr[i], X.indptr[i + 1])
                x_indices = X.indices[span]
                row = zip(fields[x_indices], x_indices, X.data[span]) if is_ffm_format \
                    else zip(x_indices, X.data[span])
            else:
                nz = X[i] != 0
                row = zip(fields[nz], np.where(nz)[0], X[i, nz]) if is_ffm_format \
                    else zip(np.where(nz)[0], X[i, nz])

            if is_ffm_format:
                s = " ".join(value_pattern % (f, j, x) for f, j, x in row)
            else:
                s = " ".join(value_pattern % (j, x) for j, x in row)

            if y_is_sp:
                labels_str = label_pattern % y.data[i]
            else:
                labels_str = label_pattern % y[i]

            f_handle.write((line_pattern % (labels_str, s)).encode('ascii'))

class BaseXLearnModel(BaseEstimator):
    """ Implementation of Scikit-learn interface for xlearn models.

    :param model_type: one of 'lr', 'fm', 'ffm'
    :param task: 'binary' for classification or 'reg' for regression
    :param metric: 'acc', 'prec', 'recall', 'f1', 'auc' for classification,
        and 'mae', 'mape', 'rmsd (rmse)' for regression.
    :param log: location of log
    :param lr: learning rate
    :param k: latent factor for factorization
    :param reg_lambda: alias for lambda
    :param init: initial value
    :param fold: number of fold used in cross validation
    :param epoch: number of training epoch
    :param stop_window: window size for early stopping
    :param opt: optimizer option, one of 'sgd', 'adagrad', 'ftrl'
    :param nthread: number of threads (Deprecated, please use n_jobs)
    :param n_jobs: number of threads used to run xlearn.
    :param block_size: block size for on-disk training.
    :param alpha: alpha for FTRL
    :param beta: beta for FTRL
    :param lambda_1: lambda_1 for FTRL
    :param lambda_2: lambda_2 for FTRL
    :param kwargs: extra input arguments
    """
    
    def __init__(self, model_type='fm', task='binary', metric='auc', block_size=500,
                 lr=0.2, k=4, reg_lambda=0.1, init=0.1, fold=1, epoch=5, stop_window=2,
                 opt='sgd', nthread=None, n_jobs=4, alpha=1, beta=1, lambda_1=1, lambda_2=1,
                 **kwargs):
        self.model_type = model_type
        self.task = task
        self.metric = metric
        self.lr = lr
        self.k = k
        self.reg_lambda = reg_lambda
        self.init = init
        self.fold = fold
        self.epoch = epoch
        self.opt = opt
        self.nthread = nthread
        self.n_jobs = n_jobs
        self.stop_window = stop_window
        self.alpha = alpha
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.kwargs = kwargs
        self.block_size = block_size

        # initialize internal structure
        self._XLearnModel = None
        self._temp_model_file = tempfile.NamedTemporaryFile(delete=True)
        self._temp_weight_file = tempfile.NamedTemporaryFile(delete=True)
        self.weights = None
        self.fields = None

    def get_model(self):
        """ Return internal XLearn model.

        This will raise exception when model is not fitted

        :return: the underlying XLearn model
        """
        if self._XLearnModel is None:
            raise Exception('Need to call fit beforehand')

        return self._XLearnModel

    def get_params(self, deep=False):
        """ Get model parameters

        :param deep: is deep copy
        :return: model parameters
        """
        params = super(BaseXLearnModel, self).get_params(deep=deep)

        return params

    def get_xlearn_params(self):
        """ Get xlearn model parameters

        :return: model parameters used for training
        """
        xlearn_param = self.get_params()

        # rename reg_lambda as lambda, and remove model_type
        xlearn_param['lambda'] = xlearn_param.pop('reg_lambda')
        xlearn_param.pop('model_type')

        # rename n_jobs to nthread for _XLearnModel
        n_jobs = xlearn_param.pop('n_jobs')
        if 'nthread' in xlearn_param and xlearn_param['nthread'] is not None:
            warnings.warn("The nthread parameter has been deprecated. Please use n_jobs instead."
                          , DeprecationWarning)
        else:
            xlearn_param['nthread'] = n_jobs

        return xlearn_param

    def fit(self, X, y=None, fields=None,
            is_lock_free=True, is_instance_norm=True, 
            eval_set=None, is_quiet=False):
        """ Fit the XLearn model given feature matrix X and label y

        :param X: array-like or a string specifying file location
                  Feature matrix
        :param y: array-like
                  Label
        :param fields: array-like
                  Fields for FFMModel. Default as None
        :param is_lock_free: is using lock-free training
        :param is_instance_norm: is using instance-wise normalization
        :param eval_set: a 2-element list representing (X_val, y_val) or a string specifying file location
        :param is_quiet: is training model quietly
        """

        if self.model_type == 'fm':
            self._XLearnModel = create_fm()
        elif self.model_type == 'lr':
            self._XLearnModel = create_linear()
        elif self.model_type == 'ffm':
            self._XLearnModel = create_ffm()
        else:
            raise Exception('model_type must be fm, ffm or lr')

        # create temporary files for training data
        temp_train_file = tempfile.NamedTemporaryFile(delete=True)

        if y is None:
            assert isinstance(X, str), 'X must be a string specifying training file location' \
                                       ' when only X specified'
            self._XLearnModel.setTrain(X)

        else:
            X, y = check_X_y(X, y, accept_sparse=['csr'], y_numeric=True, multi_output=False)

            if self.model_type == 'ffm':
                assert fields is not None, 'Must specify fields in FFMModel'
                self.fields = fields

            # convert data into libsvm/libffm format for training
            # TODO: replace conversion with DMatrix
            self._convert_data(X, y, temp_train_file.name, fields=self.fields)
            self._XLearnModel.setTrain(temp_train_file.name)

        # TODO: find out what task need to set sigmoid
        if self.task == 'binary':
            self._XLearnModel.setSigmoid()

        # set lock-free, is quiet training and instance-wise normalization
        if not is_lock_free:
            self._XLearnModel.disableLockFree()

        if is_quiet:
            self._XLearnModel.setQuiet()

        if not is_instance_norm:
            if self.model_type in ['fm', 'ffm']:
                self._XLearnModel.disableNorm()
            else:
                warnings.warn('Setting is_instance_norm to False is ignored. It only applies to fm or ffm.')

        params = self.get_xlearn_params()

        # check if validation set exists or not
        if eval_set is not None:
            if isinstance(eval_set, str):
                self._XLearnModel.setValidate(eval_set)
            else:
                if not (isinstance(eval_set, list) and len(eval_set) == 2):
                    raise Exception('eval_set must be a 2-element list')

                # extract validation data
                X_val, y_val = check_X_y(eval_set[0], eval_set[1], 
                    accept_sparse=['csr'], 
                    y_numeric=True, 
                    multi_output=False)

                temp_val_file = tempfile.NamedTemporaryFile(delete=True)
                self._convert_data(X_val, y_val, temp_val_file.name, fields=self.fields)
                self._XLearnModel.setValidate(temp_val_file.name)

        # set up files for storing weights
        self._XLearnModel.setTXTModel(self._temp_weight_file.name)

        # fit model
        self._XLearnModel.fit(params, self._temp_model_file.name)

        # acquire weights
        self._parse_weight(self._temp_weight_file.name)

        # remove temporary files for training
        self._remove_temp_file(temp_train_file)

    def predict(self, X):
        """ Generate prediction using feature matrix X

        :param X: array-like
                  Feature matrix
        :return: prediction
        """

        # convert data to libsvm or libffm format
        temp_test_file = tempfile.NamedTemporaryFile(delete=True)

        if isinstance(X, str):
            self._XLearnModel.setTest(X)
        else:
            X = check_array(X, accept_sparse=['csr'])
            self._convert_data(X, None, temp_test_file.name, fields=self.fields)
            self._XLearnModel.setTest(temp_test_file.name)

        # generate output
        temp_output_file = tempfile.NamedTemporaryFile(delete=True)
        self.get_model().predict(self._temp_model_file.name, temp_output_file.name)

        # read output into numpy
        pred = np.loadtxt(temp_output_file.name)

        # remove temporary test data and output file
        self._remove_temp_file(temp_test_file)
        self._remove_temp_file(temp_output_file)

        return pred

    def feature_importance_(self):
        """TODO: analyze weight matrix to get feature importance"""
        pass

    def _convert_data(self, X, y, filepath, fields=None):
        """
        Convert feature matrix X and label y into libsvm format
        :param X: feature matrix
        :param y: label or None
        """
        if y is None:
            # create dummy label for test
            y = np.zeros(X.shape[0], dtype=np.int8)

        if fields is not None:
            # check if the model is ffm or not
            if self.model_type is not 'ffm':
                fields = None
                warnings.warn('Fields are ignored as it is not FFMModel')
            else:
                assert X.shape[1] == len(fields)

        try:
            write_data_to_xlearn_format(X, y, filepath, fields=fields)
        except:
            raise Exception('Failed to convert feature matrix X and label y to xlearn data format')

    def _parse_weight(self, file_name):
        # estimate number of features from txt file
        num_lines = sum(1 for line in open(file_name))
        num_features = int((num_lines-1)/2)

        if self.model_type in ['fm', 'ffm']:
            weight_vec = np.genfromtxt(file_name, skip_footer=num_features)
            weight_mtx = np.genfromtxt(file_name, skip_header=num_features+1)
        else:
            weight_vec = np.genfromtxt(file_name)
            weight_mtx = None

        self.weights = (weight_vec, weight_mtx)

    def _remove_temp_file(self, temp_file):
        # The temp_file might be converted to binary file during training/inference.
        # remove both original temp_file and derived binary file if exist
        temp_bin_file = temp_file.name + '.bin'
        if os.path.exists(temp_bin_file):
            os.remove(temp_bin_file)
        temp_file.close()

    def __delete__(self, instance):
        self._temp_model_file.close()
        self._temp_weight_file.close()

class FMModel(BaseXLearnModel):
    """ Factorization machine (FM) model
    """
    def __init__(self, model_type='fm', task='binary', metric='auc', block_size=500,
                 lr=0.2, k=4, reg_lambda=0.1, init=0.1, fold=1, epoch=5, stop_window=2,
                 opt='sgd', nthread=None, n_jobs=4, alpha=1, beta=1, lambda_1=1, lambda_2=1,
                 **kwargs):
        assert model_type == 'fm', 'Model type must be set to fm for FMModel'
        super(FMModel, self).__init__(model_type, task, metric, block_size,
                                      lr, k, reg_lambda, init, fold, epoch, stop_window,
                                      opt, nthread, n_jobs, alpha, beta, lambda_1, lambda_2,
                                      **kwargs)

    def __delete__(self, instance):
        super(FMModel, self).__delete__(instance)

class LRModel(BaseXLearnModel):
    """ linear model
    """
    def __init__(self, model_type='lr', task='binary', metric='auc', block_size=500,
                 lr=0.2, k=4, reg_lambda=0.1, init=0.1, fold=1, epoch=5, stop_window=2,
                 opt='sgd', nthread=None, n_jobs=4, alpha=1, beta=1, lambda_1=1, lambda_2=1,
                 **kwargs):
        assert model_type == 'lr', 'Model type must be set to lr for LRModel'
        super(LRModel, self).__init__(model_type, task, metric, block_size,
                                      lr, k, reg_lambda, init, fold, epoch, stop_window,
                                      opt, nthread, n_jobs, alpha, beta, lambda_1, lambda_2,
                                      **kwargs)

    def __delete__(self, instance):
        super(LRModel, self).__delete__(instance)

class FFMModel(BaseXLearnModel):
    """ Field-aware factorization machine (FFM) model
    """
    def __init__(self, model_type='ffm', task='binary', metric='auc', block_size=500,
                 lr=0.2, k=4, reg_lambda=0.1, init=0.1, fold=1, epoch=5, stop_window=2,
                 opt='sgd', nthread=None, n_jobs=4, alpha=1, beta=1, lambda_1=1, lambda_2=1,
                 **kwargs):
        assert model_type == 'ffm', 'Model type must be set to ffm for FFMModel'
        super(FFMModel, self).__init__(model_type, task, metric, block_size,
                                       lr, k, reg_lambda, init, fold, epoch, stop_window,
                                       opt, nthread, n_jobs, alpha, beta, lambda_1, lambda_2,
                                       **kwargs)

    def __delete__(self, instance):
        super(FFMModel, self).__delete__(instance)
