import os,joblib,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

df=pd.read_csv("data/earthquake_warning_data.csv"); X=df.drop(columns=["warning"]); y=df.warning
pipe=Pipeline([("feature_selection",SelectKBest(f_classif,k=5)),
("classifier",RandomForestClassifier(n_estimators=250,max_depth=12,random_state=42,n_jobs=-1,class_weight="balanced"))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
m={"Accuracy":accuracy_score(yte,p),"Precision":precision_score(yte,p,zero_division=0),
"Recall":recall_score(yte,p,zero_division=0),"F1":f1_score(yte,p,zero_division=0)}
os.makedirs("models",exist_ok=True); joblib.dump(pipe,"models/earthquake_warning_model.joblib")
with open("models/metrics.txt","w") as f:
    [f.write(f"{k}={v:.4f}\n") for k,v in m.items()]
    f.write("Selected features="+", ".join(X.columns[pipe.named_steps["feature_selection"].get_support()])+"\n")
print(m)
