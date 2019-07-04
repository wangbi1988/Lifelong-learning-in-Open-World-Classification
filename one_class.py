# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:50:30 2018

@author: WangBi
"""


from DataSet import DataSet;
import h5py
from scipy.sparse import csr_matrix as csr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

def load_mnist(src):
    file = h5py.File('{}_encoder.h5'.format(src), 'r');
    train_x = file['train_x'][:];
    train_y = np.transpose(np.array([file['train_y'][:]]));
    train = DataSet(train_x, train_y, onehot = True);
    
    validation_x = file['validation_x'][:];
    validation_y = np.transpose(np.array([file['validation_y'][:]]));
    validation = DataSet(validation_x, validation_y, onehot = True);
    
    test_x = file['test_x'][:];
    test_y = np.transpose(np.array([file['test_y'][:]]));
    test = DataSet(test_x, test_y, onehot = True);
    
    file.close();
    return {'train':train, 'validation':validation, 'test':test};


def composer(dict_data, sub_index, offset = 0):
    x = dict_data['train'].data[sub_index['train'], :] + offset;
    y = dict_data['train'].labels[sub_index['train'], :];
    train = DataSet(x, y, onehot = False);
    x = dict_data['validation'].data[sub_index['validation'], :] + offset;
    y = dict_data['validation'].labels[sub_index['validation'], :];
    validation = DataSet(x, y, onehot = False);
    x = dict_data['test'].data[sub_index['test'], :] + offset;
    y = dict_data['test'].labels[sub_index['test'], :];
    test = DataSet(x, y, onehot = False);
    return {'train':train, 'validation':validation, 'test':test};


dict_data = load_mnist('mnist');

train = dict_data['train'];
validation = dict_data['validation'];
test = dict_data['test'];

train_labels = np.argmax(train.labels, 1);
validation_labels = np.argmax(validation.labels, 1);
test_labels = np.argmax(test.labels, 1);

sub_train_index = np.less_equal(train_labels, 5); # <= 4
sub_validation_index = np.less_equal(validation_labels, 5); # <= 4
sub_test_index = np.less_equal(test_labels, 5); # <= 4

X = composer(dict_data, {'train':sub_train_index, 
                                  'validation':sub_validation_index,
                                  'test':sub_test_index});
X_train = X['train'].data;
X_test = X['test'].data;

sub_train_index = np.equal(train_labels, 9); # <= 4
sub_validation_index = np.equal(validation_labels, 9); # <= 4
sub_test_index = np.equal(test_labels, 9); # <= 4

X_ = composer(dict_data, {'train':sub_train_index, 
                                  'validation':sub_validation_index,
                                  'test':sub_test_index});
X_outliers = X_['test'].data;
#xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
#X = 0.3 * np.random.randn(100, 2)
#X_train = np.r_[X + 2, X - 2]
## Generate some regular novel observations
#X = 0.3 * np.random.randn(20, 2)
#X_test = np.r_[X + 2, X - 2]
## Generate some abnormal novel observations
#X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", degree = 6)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

print('train_error is {}, test_error is {}, outliers_error is {}'.format(
        n_error_train / len(X_train), n_error_test / len(X_test), n_error_outliers / len(X_outliers)));
# plot the line, the points, and the nearest vectors to the plane
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)

#plt.title("Novelty Detection")
#plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
#a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
#plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

#s = 40
#b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
#b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
#                 edgecolors='k')
#c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
#                edgecolors='k')
#plt.axis('tight')
#plt.xlim((-5, 5))
#plt.ylim((-5, 5))
#plt.legend([a.collections[0], b1, b2, c],
#           ["learned frontier", "training observations",
#            "new regular observations", "new abnormal observations"],
#           loc="upper left",
#           prop=matplotlib.font_manager.FontProperties(size=11))
#plt.xlabel(
#    "error train: %d/200 ; errors novel regular: %d/40 ; "
#    "errors novel abnormal: %d/40"
#    % (n_error_train, n_error_test, n_error_outliers))
#plt.show()