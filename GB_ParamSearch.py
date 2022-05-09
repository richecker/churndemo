# Setup
import matplotlib
matplotlib.use('Agg')
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle

#Adding First Comment here

#Adding Second Comment here

#Adding Third Comment

#Adding Fourth Comment

#adding fifth comment

#Adding sixth comment

#capture inputs
loss_ = sys.argv[1]
n_estimators_ = int(sys.argv[2])


# Import Data
df = pd.read_csv('/mnt/data/customer_churn_data/prepared.csv', header=0, index_col=0)
columns = list(df.columns)
columns.remove('churn_Y')
y = df["churn_Y"].values
X = df[columns].values

# baseline data
df_baseline = pd.read_csv('/mnt/data/customer_churn_data/churndata.csv', header=0, index_col=0)
create_dummies = pd.get_dummies(df_baseline['churn'],prefix='churn', drop_first=True)  
df_baseline=pd.concat([df_baseline,create_dummies],axis=1)
df_baseline = df_baseline.drop(['churn'], axis=1)
baseline_cols = ['consecmonths', 'KPI_DeviceSat', 'KPI_NetworkQual']
y_baseline = df_baseline['churn_Y'].values
X_baseline = df_baseline[baseline_cols].values

# Build and Evaluate New Model
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(loss = loss_, n_estimators = n_estimators_, learning_rate = 1)
clf = clf.fit(X, y)
clfprb = clf.predict_proba(X)

from sklearn import metrics
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

tprs = []
aucs = []
accs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in kf.split(X):
    probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
    preds_ = clf.fit(X[train], y[train]).predict(X[test])
    fpr, tpr, thresholds = metrics.roc_curve(y[test], probas_[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    accuracy = metrics.accuracy_score(y[test], preds_)
    accs.append(accuracy)
    i += 1
    
mean_acc = np.mean(accs)
std_acc = np.std(accs)
mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

#Get Baseline Model

baseline_model = pickle.load(open('/mnt/results/baseline.pkl', 'rb'))
tprs_baseline = []
aucs_baseline = []
accs_baseline = []
mean_fpr_baseline = np.linspace(0, 1, 100)
 
probas_baseline = baseline_model.predict_proba(X_baseline)
preds_baseline = baseline_model.predict(X_baseline)
fpr_baseline, tpr_baseline, thresholds_baseline = metrics.roc_curve(y_baseline, probas_baseline[:, 1])
tprs_baseline.append(np.interp(mean_fpr_baseline, fpr_baseline, tpr_baseline))
tprs[-1][0] = 0.0
roc_auc_baseline = metrics.auc(fpr_baseline, tpr_baseline)
aucs_baseline.append(roc_auc_baseline)
accuracy_baseline = metrics.accuracy_score(y_baseline, preds_baseline)
accs_baseline.append(accuracy_baseline)
acc_baseline = np.mean(accs_baseline)
auc_baseline = np.mean(aucs_baseline)

# Make ROC Plot
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8, label='New: Avg AUC = %0.3f Avg Acc = %0.3f' % (mean_auc, mean_acc))
plt.plot(fpr_baseline,tpr_baseline,color='red', label=        'Baseline: AUC = %0.3f Acc = %0.3f' % (auc_baseline, acc_baseline))
#plt.text(0.6, 0.125, 'Mean AUC = %0.3f' % (mean_auc))
#plt.text(0.6, 0.2, 'Mean Accuracy = %0.3f' % (mean_acc))
 
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + 2*std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - 2*std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 2 std. dev.')
 
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("5-Fold CV ROC Curve and Metrics: Loss = " + loss_ + ", N_Estimators = " + str(n_estimators_))
plt.legend(loc="lower right")
 
#write out resutls
plt.savefig('/mnt/results/AUC_ACC_' + loss_ + '_' + str(n_estimators_) + '.png', format='png')
plt.savefig('/mnt/results/AUC_ACC.png', format='png')
plt.gcf().clear()

#make confusion matrix plot
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig('/mnt/results/ConfMatx_' + loss_ + '_' + str(n_estimators_) + '.png', format='png')
    plt.savefig('/mnt/results/ConfMatx_New.png', format='png')
    plt.gcf().clear()
    
plot_confusion_matrix(cm           = metrics.confusion_matrix(y[test], preds_), 
                      normalize    = False,
                      target_names = ['no churn', 'churn'],
                      title        = "Confusion Matrix for Loss = " + loss_ + ", N_Estimators = " + str(n_estimators_))
                      
import json
 
with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"Acc": round(mean_acc, 3), "AUC": round(mean_auc, 3)}))
