import os
import shutil
import tempfile
import numpy as np

from .xlearn import create_linear, create_fm, create_ffm
from sklearn.base import BaseEstimator
from sklearn.datasets import dump_svmlight_file

class BaseXLearnModel(BaseEstimator):
    """ Implementation of Scikit-learn interface for xlearn models.
    """
    def __init__(self, model_type='fm', task='binary', metric='auc',
                        lr=0.2, k =4, reg_lambda=0.1, init=0.1, fold=1, epoch=5,
                        opt='sgd', nthread=4, alpha=1, beta=1, lambda_1=1, lambda_2=1,
                        **kwargs):
        """
        :param model_type: one of 'lr', 'fm', 'ffm'
        :param task: 'binary'
        :param metric: 'auc'
        :param log: location of log
        :param lr: learning rate
        :param k: latent factor for factorization
        :param reg_lambda: alias for lambda
        :param init: initial value (TODO: check this parameter)
        :param fold: number of fold used in cross validation
        :param epoch: number of training epoch
        :param opt: optimizer option, one of 'sgd', 'adam', 'ftrl'
        :param nthread: number of threads
        :param alpha: alpha for FTRL
        :param beta: beta for FTRL
        :param lambda_1: lambda_1 for FTRL
        :param lambda_2: lambda_2 for FTRL
        :param kwargs: extra input arguments
        """

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
        self.alpha = alpha
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.kwargs = kwargs

        # initialize internal structure
        self._XLearnModel = None
        self._temp_model_file = tempfile.NamedTemporaryFile(delete=True)

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

        # rename reg_lambda as lambda, and remove model_type
        params['lambda'] = params.pop('reg_lambda')
        params.pop('model_type')
        return params

    def fit(self, X, y):
        """ Fit the XLearn model given feature matrix X and label y
        :param X: array-like
                  Feature matrix
        :param y: array-like
                  Label
        """
        if self.model_type == 'fm':
            self._XLearnModel = create_fm()
        elif self.model_type == 'lr':
            self._XLearnModel = create_linear()
        elif self.model_type == 'ffm':
            self._XLearnModel = create_ffm()
        else:
            raise Exception('model_type must be fm, ffm or lr')

        #TODO: find out what task need to set sigmoid
        if self.task == 'binary':
            self._XLearnModel.setSigmoid()

        params = self.get_params(deep=True)

        # create temporary files for training data
        temp_train_file = tempfile.NamedTemporaryFile(delete=True)

        # convert data into libsvm format for training
        self._convert_data(X, y, temp_train_file.name)
        self._XLearnModel.setTrain(temp_train_file.name)

        # fit model
        self._XLearnModel.fit(params, self._temp_model_file.name)

        # remove temporary files for training
        self._remove_temp_file(temp_train_file)

    def predict(self, X):
        """ Generate prediction using feature matrix X

        :param X: array-like
                  Feature matrix
        :return: prediction
        """
        # convert data to libsvm format
        temp_test_file = tempfile.NamedTemporaryFile(delete=True)
        self._convert_data(X, None, temp_test_file.name)
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

    def _convert_data(self, X, y, filepath):
        """
        Convert feature matrix X and label y into libsvm format
        :param X: feature matrix
        :param y: label or None
        """
        if y is None:
            # create dummy label for test
            y = np.zeros(X.shape[0], dtype=np.int8)

        try:
            dump_svmlight_file(X, y, filepath)
        except:
            raise Exception('failed to convert feature matrix X and label y to libsvm format')

    def _remove_temp_file(self, temp_file):
        # The temp_file might be converted to binary file during training/inference.
        # remove both original temp_file and derived binary file if exist
        temp_bin_file = temp_file.name + '.bin'
        if os.path.exists(temp_bin_file):
            os.remove(temp_bin_file)
        temp_file.close()

    def __delete__(self, instance):
        self._temp_model_file.close()


class FMModel(BaseXLearnModel):
    def __init__(self, model_type='fm', task='binary', metric='auc',
                        lr=0.2, k =4, reg_lambda=0.1, init=0.1, fold=1, epoch=5,
                        opt='sgd', nthread=4, alpha=1, beta=1, lambda_1=1, lambda_2=1,
                        **kwargs):
        assert model_type == 'fm', 'Model type must be set to fm for FMModel'
        super(FMModel, self).__init__(model_type, task, metric,
                                      lr, k, reg_lambda, init, fold, epoch,
                                      opt, nthread, alpha, beta, lambda_1, lambda_2,
                                      **kwargs)

    def __delete__(self, instance):
        super(FMModel, self).__delete(instance)