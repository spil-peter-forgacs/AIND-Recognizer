import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
        logL: loglikelihood
        N: the sample size of the training set (number of observations / data points)
        p: the total number of free parameters (model degrees of freedom).
            p1 = transistion_matrix_probabilities(=n*n) + gaussian_means(=n*d) + gaussian_variance(=n*d) = n*n + n*d + n*d = n*n + 2*d*n
            p2 = n*(n-1) + 2*d*n
            p3 = n*(n-1) + (n-1) + 2*d*n = n*n + 2*d*n - 1
                n: the number of model states (HMM states)
                d: the number of features or data points that is used to train the model: len(self.X[0])
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_bic = None
        best_hmm_model = None

        # Check the different components.
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = hmm_model.score(self.X, self.lengths)
                logN = np.log(len(self.words))
                d = len(self.X[0])
                p = n*n + 2*d*n - 1
                bic = -2 * logL + p * logN

                if (best_bic is None) or (best_bic > bic):
                    best_bic = bic
                    best_hmm_model = hmm_model
            except:
                pass

        return best_hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    DIC = log(P(original world)) - average(log(P(otherwords)))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:

            best_dic = None
            best_hmm_model = None

            word_number = (len(self.words) - 1)

            # Check the different components.
            for n in range(self.min_n_components, self.max_n_components + 1):
                if len(self.X) >= n:
                    try:
                        # Calculate the logL for the given word.
                        hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                        logL = hmm_model.score(self.X, self.lengths)

                        # Calculate the logLs for the other words.
                        logL_others = 0;
                        for word in self.hwords:
                            if word != self.this_word:
                                X_others, lengths_others = self.hwords[word]
                                if len(X_others) >= n:
                                    try:
                                        hmm_model_others = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X_others, lengths_others)
                                        logL_others = logL_others + hmm_model_others.score(X_others, lengths_others)
                                    except:
                                        pass

                        average = logL_others / word_number
                        dic = logL - average

                        if (best_dic is None) or (best_dic < dic):
                            best_dic = dic
                            best_hmm_model = hmm_model
                    except:
                        pass

            return best_hmm_model

        except Exception as e:
            print(str(e))

            if self.verbose:
                print("failure on {}".format(self.this_word))

            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV

        try:

            best_logL = None
            best_num_components = 3

            # If there are less, than 3 examples, return the model with 3 components.
            # As we don't want value exception.
            min_examples = 3
            if len(self.lengths) < min_examples:
                hmm_model = self.base_model(best_num_components)
                return hmm_model

            # We have enough examples. We can use KFold.

            # Check the different components.
            for i in range(self.min_n_components, self.max_n_components + 1):

                # Kfold.
                split_method = KFold()
                for train_index, test_index in split_method.split(self.lengths):

                    train_X, train_lengths = combine_sequences(train_index, self.sequences)
                    #train_lengths = self.lengths[train_index]
                    test_X, test_lengths = combine_sequences(test_index, self.sequences)
                    #test_lengths = self.lengths[test_index]

                    hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(train_X, train_lengths)

                    logL = hmm_model.score(test_X, test_lengths)

                    if (best_logL is None) or (best_logL < logL):
                        best_logL = logL
                        best_num_components = i

            if self.verbose:
                print("model created for {} with {} components".format(self.this_word, best_num_components))

            hmm_model = self.base_model(best_num_components)
            return hmm_model

        except:

            if self.verbose:
                print("failure on {}".format(self.this_word))

            return None
