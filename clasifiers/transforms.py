from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

import data_loader
import paramaters
import numpy as np

def lda_transform(train_featurs, train_labels ):

    clf = LinearDiscriminantAnalysis()

    # train_data = data_loader.Data(train=True, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)
    # test_data = data_loader.Data(train=False, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)

    train_featurs = np.array(train_featurs, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int)

    # train_labels = train_labels.tolist()

    # test_featurs = np.array(test_data.X, dtype=np.float64)
    # test_labels = np.array(test_data.Y, dtype=np.int)



    clf.fit(list(train_featurs), list(train_labels))

    transformed = clf.transform(train_featurs)

    return transformed
    # score = clf.score(test_featurs, test_labels)
    #
    # print(score)

def PCA_transform(train_featurs, train_labels):

    clf = PCA()

    train_featurs = np.array(train_featurs, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int)

    clf.fit(list(train_featurs), list(train_labels))

    transformed = clf.transform(train_featurs)

    return transformed