import pandas as pd
from numpy import ndarray
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt



def scoring(X_test: pd.DataFrame, y_test: pd.DataFrame, model, predictions: ndarray):
    score = model.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    VN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    VP = conf_matrix[1][1]
    print(f"Matriz de confusión del {model} - Test")
    print ("Verdaderos Negativos \t", VN)
    print ("Falsos Positivos \t", FP)
    print ("Falsos Negativos \t", FN)
    print ("Verdaderos Positivos \t", VP)
    #especifidad
    especifidad = VN/(VN+FP)*100
    print("Especificidad: "+ str((VN/(VN+FP))*100))
    #Sensivilidad
    sensibilidad = VP/(VP+FN)*100
    print("Sensibilidad: "+str((VP/(VP+FN))*100))
    RFP, RVP, umbrales = roc_curve(y_test, predictions)

    sns.lineplot(x=RFP, y=RVP)
    sns.lineplot(x=[0, 1], y=[0, 1], color="black", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Espesificidad\n1.0                0.8                0.6                0.4                0.2                0.0", fontsize=10)
    plt.xlabel("1 - Espesificidad = Ratio de falsos positivos")
    plt.ylabel("Sensibilidad = Ratio verdaderos positivos")
    plt.grid(True)
    print("AUC - Datos de test")
    print(roc_auc_score(y_test, predictions))
    return score, conf_matrix, especifidad, sensibilidad

