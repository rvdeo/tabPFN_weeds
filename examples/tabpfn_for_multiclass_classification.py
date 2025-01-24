#  Copyright (c) Prior Labs GmbH 2025.

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

'''
# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"))

# Predict labels
y_predict = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_predict))
'''


from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')


def calc_roc_auc(y_test, y_onehot_test, y_score,n_classes):
    
    
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    
    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")
    
    macro_roc_auc_ovr = roc_auc_score(
        y_test,
        y_score,
        multi_class="ovr",
        average="macro",
    )
    
    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")
    
   
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    
    # Average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")
    return fpr, tpr, roc_auc


def plot_roc_auc(y_onehot_test, y_score,n_classes,fpr,tpr,roc_auc):    


    fig, ax = plt.subplots(figsize=(6, 6))
    
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    
    colors = cycle(mcolors.TABLEAU_COLORS)#(["blue", "aqua", "darkorange", "cornflowerblue","gray"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
            despine=True,
        )
    
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )

    
    
def run_classifier(X,y,target_names):
    

    np.random.RandomState(0)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    #X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_predict_logres = classifier.predict(X_test)
    print("Accuracy y_predict_logres", accuracy_score(y_test, y_predict_logres))
    
    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print("Accuracy y_predict", accuracy_score(y_test, y_predict))
    
    
    y_score = clf.predict_proba(X_test)
    
    
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    #y_onehot_test.shape  # (n_samples, n_classes)
    
    fpr, tpr, roc_auc = calc_roc_auc(y_test,y_onehot_test, y_score,n_classes)
    plot_roc_auc(y_onehot_test, y_score,n_classes,fpr,tpr,roc_auc)



#iris = load_iris()
#target_names = iris.target_names
#X, y = iris.data, iris.target
#y = iris.target_names[y]

#run_classifier(X,y,target_names)


df = pd.read_csv("data/exp_325/Unbalanced_325.csv")

# Prepare features and target
X = df.drop('Species', axis=1)
y = df[['Species']]
target_names = y['Species'].unique()
run_classifier(X,y,target_names)

'''
class_of_interest = "virginica"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id



display = RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    y_score[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
    despine=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)",
)
'''





