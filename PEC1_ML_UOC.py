# Carlos Expósito Carrera, ML UOC 2023-2024/2. PEC 1

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# Cargamos los datos, dejando de banda el nombre de los atributos 
my_data = np.genfromtxt('LARGE.csv', delimiter=',', skip_header=1)
# Creamos un array para los nombres de los atributos
attribute_names = [ 'variance', 'skewness', 'curtosis', 'entropy', 'class' ]
# Cuantos ejemplos hay (shape[0] nos indica la primera dimensión del array Numpy, que equivale a los ejemplos)
sample_num = my_data.shape[0]
print("número de ejemplos: ", sample_num)
# Cuantos atributos hay (shape[1] nos indica la segunda dimensión del array Numpy, que equivale a los atributos)
attribute_num = my_data.shape[1]
print("Número de atributos: ", attribute_num)
# Se busca el índice correspondiente a class y se indican cuantas ocurrencias únicas hay
class_index = attribute_names.index('class')
class_num = len(np.unique(my_data[:,class_index]))
print("Número de clases: ", class_num)
# Se cuentan las ocurrencias de las clases, y se muestran
class_examples = np.unique(my_data[:,class_index],return_counts=True)
print ("Ejemplos de ", class_examples[0][0],": ", class_examples[1][0])
print ("Ejemplos de ", class_examples[0][1],": ", class_examples[1][1])

'''
    Se decide realizar una selección de características univariante para evaluar la función de bonanza de las características.
    Posteriormente, analizaremos si alguna de las variables puede ser eliminada. En cualquier caso, una vez decididas las características, 
    normalizaremos los datos para su posterior tratamiento.
'''
# Usaremos sklearn para ello. Crearemos un objeto SelectKBest y escogeremos las 2 mejores características
# Documentación: https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
# separamos las características de la clase
X= my_data[:, :-1]
Y= my_data[:, -1]

X_new = SelectKBest(f_classif, k=2).fit(X,Y)
feature_scores = X_new.scores_
selected_features_indices = X_new.get_support()

# Mostraremos las puntuaciones del algoritmo, y las características seleccionadas.
selected_features = np.array(attribute_names[:-1])[selected_features_indices]
print("Puntuaciones de las características:")
for feature, score in zip(attribute_names[:-1], feature_scores):
    print(f"{feature}: {score}")

print("\nCaracterísticas seleccionadas:")
for feature in selected_features:
    print(feature)
# Finalmente, transformamos el array y comprobamos que las dimensiones sean las adecuadas
my_data_transformed = X_new.transform(X)
print(my_data_transformed.shape)

# Proseguiremos con la normalización de los valores mediante su estandarización
# Véase https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
scaler.fit(my_data_transformed)
my_data_preprocessed = scaler.transform(my_data_transformed)

# Mostramos los primeros 10 casos
print(selected_features,"\n",my_data_preprocessed[0:10])

'''
    A continuación, aplicaremos el algoritmo Kmeans con dos centroides escogidos manualmente
    La elección de los centroides se hará mediante la elección aleatoria de dos índices del array a tratar
    https://stackoverflow.com/questions/43506766/randomly-select-from-numpy-array 
'''
# Escogemos dos índices
index = np.random.choice(my_data_preprocessed.shape[0], 2, replace = False)
# Escogemos los objetos asociados a los índices, que serán nuestros centroides
init_centroids= my_data_preprocessed[index,:]
# Realizamos el Kmeans para dos clusters y los centroides iniciales escogidos. Se deja n_init como auto
kmeans = KMeans(n_clusters = 2, init = init_centroids)
kmeans.fit(my_data_preprocessed)

# Se obtienen las labels (clases predichas) del algoritmo kmeans
kmeans_pred = kmeans.labels_

# Calculamos las métricas. Para ello, usaremos las metricas de sklearn
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

# Cálculo de la exactitud (accuracy)
kmeans_accuracy = accuracy_score(Y,kmeans_pred)
# Cálculo de la precisión
kmeans_precision = precision_score(Y,kmeans_pred)
# Cálculo del recall
kmeans_recall = recall_score(Y,kmeans_pred)
# Cálculo de la matriz de confusión
kmeans_conf_matrix = confusion_matrix(Y,kmeans_pred)

# Cálculo del fall-out
# https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix 
TN = kmeans_conf_matrix[0][0] 
FN = kmeans_conf_matrix[1][0] 
TP = kmeans_conf_matrix[1][1] 
FP = kmeans_conf_matrix[0][1] 
FPR = FP/(FP+TN)
#Exponemos las métricas calculadas
print("Exactitud: ", kmeans_accuracy)
print("Precisión: ", kmeans_precision)
print("Recall: ", kmeans_recall)
print("Fall-out: ", FPR)
print("Matriz de Confusión: ")
print(kmeans_conf_matrix)

'''
    Repetimos el mismo proceso pero usando Kmeans++ para la selección de los centroides iniciales.
'''

# Realizamos el Kmeans para dos clusters. Se deja todo Default (el default de init = 'k-means++')
kmeansplus = KMeans(n_clusters = 2)
kmeansplus.fit(my_data_preprocessed)

# Se obtienen las labels (clases predichas) del algoritmo kmeans
kmeansplus_pred = kmeansplus.labels_

# Calculamos las métricas. Para ello, usaremos las metricas de sklearn
kmeansplus_accuracy = accuracy_score(Y,kmeansplus_pred)
kmeansplus_precision = precision_score(Y,kmeansplus_pred)
kmeansplus_recall = recall_score(Y,kmeansplus_pred)
kmeansplus_conf_matrix = confusion_matrix(Y,kmeansplus_pred)
TN = kmeansplus_conf_matrix[0][0] 
FN = kmeansplus_conf_matrix[1][0] 
TP = kmeansplus_conf_matrix[1][1] 
FP = kmeansplus_conf_matrix[0][1] 
FPR = FP/(FP+TN)
#Exponemos las métricas calculadas
print("Exactitud: ", kmeansplus_accuracy)
print("Precisión: ", kmeansplus_precision)
print("Recall: ", kmeansplus_recall)
print("Fall-out: ", FPR)
print("Matriz de Confusión: ")
print(kmeansplus_conf_matrix)