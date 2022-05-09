import sklearn as sk
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# baseline data
df_baseline = pd.read_csv('/mnt/data/churndata.csv', header=0, index_col=0)
create_dummies = pd.get_dummies(df_baseline['churn'],prefix='churn', drop_first=True)  
df_baseline=pd.concat([df_baseline,create_dummies],axis=1)
df_baseline = df_baseline.drop(['churn'], axis=1)
baseline_cols = ['consecmonths', 'KPI_DeviceSat', 'KPI_NetworkQual']
y_baseline = df_baseline['churn_Y'].values
X_baseline = df_baseline[baseline_cols].values


#Get Baseline Model
import pickle
baseline_model = pickle.load(open('/mnt/output/baseline.pkl', 'rb'))
tprs_baseline = []
aucs_baseline = []
accs_baseline = []
mean_fpr_baseline = np.linspace(0, 1, 100)
 
probas_ = baseline_model.predict_proba(X_baseline)
preds_ = baseline_model.predict(X_baseline)
fpr_baseline, tpr_baseline, thresholds_baseline = metrics.roc_curve(y_baseline, probas_[:, 1])
tprs_baseline.append(np.interp(mean_fpr_baseline, fpr_baseline, tpr_baseline))
tprs_baseline[-1][0] = 0.0
roc_auc_baseline = metrics.auc(fpr_baseline, tpr_baseline)
aucs_baseline.append(roc_auc_baseline)
accuracy_baseline = metrics.accuracy_score(y_baseline, preds_)
accs_baseline.append(accuracy_baseline)
acc_baseline = np.mean(accs_baseline)
auc_baseline = np.mean(aucs_baseline)

# Make Plot
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.plot(fpr_baseline,tpr_baseline,color='b', label= 'Baseline: AUC = %0.3f Acc = %0.3f' % (auc_baseline, acc_baseline))

 
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve and Metrics for the Baseline Model')
plt.legend(loc="lower right")
 
#write out resutls
plt.savefig('/mnt/results/AUC_ACC_Baseline.png', format='png')
