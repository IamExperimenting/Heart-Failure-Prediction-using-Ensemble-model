### immporting required python module
import pandas as pd, warnings, numpy as np,matplotlib.pyplot as plt,joblib,shap
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_backend
from ray.util.joblib import register_ray
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score,auc,roc_curve
from configparser import ConfigParser

## reading config file
parser = ConfigParser()
parser.read('config.ini')
feature_importance_image_path = parser.get('output_path','feat_imp_img')
shap_image_path = parser.get('output_path','shap_img')
roc_image_path = parser.get('output_path','roc_curve_img')
model_output_path = parser.get('output_path','pickle_file')
target = parser.get('input_data','target_column')


class Modelling:
    def __init__(self,data):
        self.data = data
    ## data split
    def data_split(self,data):
        np.random.seed(58)
        X = self.data.drop(target,axis=1)
        y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=58,stratify=y)
        return self
    ## generating base classifier
    def base_classifier(self):
        dummy_clf = DummyClassifier(strategy="stratified")
        dummy_clf.fit(self.X_train, self.y_train)
        score = dummy_clf.score(self.X_test, self.y_test)
        print('Base Classifier Accuracy is {:.2f}'.format(score))
    ## generating feature importance
    def feature_importance(self):
        clf_feat = RandomForestClassifier(n_estimators = 100)
        clf_feat.fit(self.X_train, self.y_train)
        features = self.X_train.columns
        importances = clf_feat.feature_importances_
        indices = np.argsort(importances)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig(feature_importance_image_path)

        row_to_show = 5
        data_for_prediction = self.X_test.iloc[row_to_show] 
        explainer = shap.TreeExplainer(clf_feat)
        shap_values = explainer.shap_values(data_for_prediction)
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction,show=False,matplotlib=True).savefig(shap_image_path)

        clf = SelectFromModel(RandomForestClassifier(n_estimators = 100),threshold=0.10)
        clf.fit(self.X_train, self.y_train)
        self.selected_feat= self.X_train.columns[(clf.get_support())]
    ## training ensemble model
    def ensemble_model(self):
        clf1 = svm.SVC(probability=True)
        clf2 = RandomForestClassifier(n_estimators=100)
        clf3 = GaussianNB()
        lr = LogisticRegression()
        sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                                use_probas=True,
                                average_probas=False,
                                meta_classifier=lr)
        
        pipe = Pipeline([
            ("selector", ColumnTransformer([
                ("selector", "passthrough", self.selected_feat)
            ], remainder="drop")),
            ('scale', StandardScaler()),
            ('ensemble_model',sclf)
        ])

        ## distributed training ##
        register_ray()
        with parallel_backend("threading",n_jobs=4):    
            pipe.fit(self.X_train,self.y_train)

        joblib.dump(pipe,model_output_path)
        y_pred = pipe.predict(self.X_test)

        # AUC ROC Curve values is considered much more important than the accuracy to evaluate the model
        predicting_probabilites = pipe.predict_proba(self.X_test)[:,1]
        fpr,tpr,thresholds = roc_curve(self.y_test,predicting_probabilites)
        plt.figure(figsize=(14,12))
        plt.subplot(222)
        plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
        plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
        plt.legend(loc = "best")
        plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)
        plt.savefig(roc_image_path)
        
        print('Accuracy of an ensemble model:{:.2f}'.format(accuracy_score(self.y_test, y_pred)))
