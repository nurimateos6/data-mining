import pandas as pd
from numpy import ndarray
from sklearn import model_selection
from sklearn.metrics import confusion_matrix


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
    return score, conf_matrix, especifidad, sensibilidad