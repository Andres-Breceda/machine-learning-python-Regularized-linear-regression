
# Se han recopilado datos socio demográficos y de recursos de salud por condado en los Estados Unidos y queremos descubrir si existe alguna relación entre los recursos sanitarios y los datos socio demográficos. 
# Para ello, es necesario que establezcas una variable objetivo (relacionada con la salud) para llevar a cabo el análisis.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv")
datacore = data.copy()

data.head()


# Para mostrar *todas* las columnas en la consola:
print(list(data.columns))

datacore = data.drop(columns=[x for x in data.select_dtypes(include="object")])

X = datacore.drop(columns=["anycondition_prevalence"])  # variables predictoras
y = datacore["anycondition_prevalence"] 
#X= sm.add_constant(X)
Y= datacore["anycondition_prevalence"] #Varibale objetivo
#print(X.dtypes)
#print(Y.dtypes)
model= sm.OLS(Y, X).fit() 
print(model.summary())

# Extraer el resumen del modelo como un DataFrame
summary_df = model.summary2().tables[1]  # La tabla 1 tiene coeficientes, p-valores, etc.

# Filtrar variables con p-value < 0.05
significant_vars = summary_df[summary_df["P>|t|"] < 0.05]

# Mostrar variables significativas
print("Variables significativas (p < 0.05):")
print(significant_vars)

#Opcioon 2 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

est = LinearRegression()
selector = RFE(est, n_features_to_select=10)  # Cambia el número si quieres más o menos variables
selector = selector.fit(X, Y)

# Ver variables seleccionadas
selected_columns = X.columns[selector.support_]
print("Variables seleccionadas:", selected_columns)

#Vemos que ambos metodos precentan resultados diferentes por lo que agregaremos los dos grupos o reduciremos las columnas del siguiente grupo


# Definir las variables de salud y las variables sociodemográficas
variables_importantes = ["anycondition_Lower 95% CI",   
"anycondition_Upper 95% CI",   
"Obesity_Upper 95% CI",       
"Heart disease_prevalence" ,   
"Heart disease_Upper 95% CI" ,
"COPD_prevalence",            
"COPD_Lower 95% CI",          
"COPD_Upper 95% CI",         
"diabetes_prevalence",         
"diabetes_Lower 95% CI",      
"diabetes_Upper 95% CI",
'Active General Surgeons per 100000 Population 2018 (AAMC)',
'Family Medicine/General Practice Primary Care (2019)']

variables_sociodemograficas = ['0-9 y/o % of total pop', '10-19 y/o % of total pop',
       '20-29 y/o % of total pop', '30-39 y/o % of total pop',
       '40-49 y/o % of total pop', '50-59 y/o % of total pop',
       '60-69 y/o % of total pop', '70-79 y/o % of total pop',
       '80+ y/o % of total pop', '% Black-alone']

# Combinar las dos listas de columnas
variables = variables_importantes + variables_sociodemograficas


# Crear DataFrame filtrado con las variables seleccionadas
df_corr = data[variables]
df_corr["anycondition_prevalence"] = data["anycondition_prevalence"]
# Calcular matriz de correlación
matriz_corr = df_corr.corr()

# Crear el gráfico de correlación con Seaborn
plt.figure(figsize=(20, 18))  # Ajusta el tamaño de la figura
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Añadir título y ajustar visualización
plt.title("Matriz de Correlación entre las Variables de Salud y Socio-Demográficas")
plt.tight_layout()
plt.show()

#En modelos lineales buscamos no tener multicolinealidad entre nuestras variables. Por lo que reduciremos el numero de variables redundantes y crearemos otra matriz de correlacion

# Crear nuevas columnas agrupadas en el DataFrame original
data["Edad_0_19"] = data["0-9 y/o % of total pop"] + data["10-19 y/o % of total pop"]
data["Edad_20_59"] = (
    data["20-29 y/o % of total pop"] +
    data["30-39 y/o % of total pop"] +
    data["40-49 y/o % of total pop"] +
    data["50-59 y/o % of total pop"]
)
data["Edad_60-80+"] = (
    data["60-69 y/o % of total pop"] +
    data["70-79 y/o % of total pop"] +
    data["80+ y/o % of total pop"]
)

# Crear un nuevo DataFrame solo con las variables agrupadas y % Black-alone
variables_sociodemograficas_agrupadas = ["Edad_60-80+", "Edad_20_59", "% Black-alone","% White-alone","% NA/AI-alone","% Asian-alone"]
df_sociodemografico = data[variables_sociodemograficas_agrupadas]

# Combinar en columnas (axis=1)
df_corr2 = pd.concat([ data[["anycondition_prevalence"]],  data[variables_importantes], df_sociodemografico], axis=1)

#Eliminar columnas
df_corr2 = df_corr2.drop(columns=['anycondition_Lower 95% CI','anycondition_Upper 95% CI','Obesity_Upper 95% CI','COPD_Lower 95% CI', 'COPD_Upper 95% CI','Heart disease_Upper 95% CI','diabetes_Lower 95% CI', 'diabetes_Upper 95% CI'])
matriz_corr = df_corr2.corr()

# Crear el gráfico de correlación con Seaborn
plt.figure(figsize=(12, 10))  # Ajusta el tamaño de la figura
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Añadir título y ajustar visualización
plt.title("Matriz de Correlación entre las Variables de Salud y Socio-Demográficas")
plt.tight_layout()
plt.show()

df_corr2.columns

#Como podemos ver existe una relacion entre las personas de raza negra o afroamericanas, tienen mas prevalencia de padecer una enfermedad como diabetes.

#Al igual podemos ver que personas con tercera edad o mas pueden tener mas prevalencia a diabetes o una enferemdad relacionada al corazon.

#como podemos ver no existe tanta relacion en edades mas jovenes con la prevalencia de diabetes o enfermedades relacionadas al corazon. Por lo que solo nos enfocaremos en personas de mas de 20 años.

#GRAFICOS DE INFORMACION POR EDAD, Y ETNIA


import matplotlib.pyplot as plt
import seaborn as sns

# Lista de variables
enfermedades = ['Heart disease_prevalence', 'COPD_prevalence', 'diabetes_prevalence']
demograficos = ['% Black-alone', '% White-alone', '% NA/AI-alone', '% Asian-alone', 'Edad_60-80+', 'Edad_20_59']

# Preparar figura
fig, axes = plt.subplots(3, 6, figsize=(24, 12))  # 3 filas × 6 columnas

# Plano lineal de ejes para fácil iteración
axes = axes.flatten()

# Contador de subplot
i = 0

# Crear cada gráfico
for enfermedad in enfermedades:
    for demografico in demograficos:
        sns.regplot(
            x=data[demografico],
            y=data[enfermedad],
            ax=axes[i],
            scatter_kws={'alpha': 0.5, 's': 10}
        )
        axes[i].set_title(f'{enfermedad} vs {demografico}', fontsize=10)
        axes[i].set_xlabel(demografico, fontsize=8)
        axes[i].set_ylabel(enfermedad, fontsize=8)
        i += 1

# Ajustar diseño
plt.tight_layout()
plt.show()


#Podemos ver que en la etnia blanca, la mayoria de las personas tiene prevalencia a problemas relacionados con el corazon.

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(5, 4))

# Colores personalizados por grupo de vecindario
colores = {
    "Brooklyn": "#1f77b4",     # Azul suave 
    "Manhattan": "#ffbf00",    # Amarillo dorado 
    "Queens": "#2ca02c",       # Verde fuerte 
    "Staten Island": "#d62728",# Rojo clásico 
    "Bronx": "#9467bd" 
}

# Gráfico de precios promedios por grupos de vecindarios
avg_price = data.groupby(["room_type", "neighbourhood_group"])["price"].mean().reset_index()
sns.barplot(
    data=avg_price,
    x="room_type",
    y="price",
    hue="neighbourhood_group",
    palette=colores,
    ax=ax
)

# Mostrar el gráfico
plt.show()

#MODELO DE REGRESION LINEAL

data_model = df_corr2
X = data_model.drop(columns=['anycondition_prevalence'])
#X =sm.add_constant(X)
Y=data_model["anycondition_prevalence"]

model = sm.OLS(Y,X)
resultados = model.fit()
resultados.summary()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

#Variables predictoras y objetivo
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
model_lineal = LinearRegression()
model_lineal.fit(X_train,y_train)

#Predicciones
y_pred_lineal = model_lineal.predict(X_test)

# Evaluar el modelo
print("Regresión Lineal (sin regularización):")
print("MSE:", mean_squared_error(y_test, y_pred_lineal))
print("R2:", r2_score(y_test, y_pred_lineal))

print(y_pred_lineal)
#Grafica de dispersion 
plt.figure(figsize=(8,6))
plt.scatter(x=y_test, y=y_pred_lineal)
plt.xlabel("Valor real (anycondition_prevalence)")
plt.ylabel("Valor prediccion del modelo")
plt.title("Prediccion Vs Real")
plt.plot([Y.min(), Y.max()],[Y.min(), Y.max()], "r--")
plt.grid(True)
plt.show()

#MODELO DE REGRESION LINEAL CON REGULARIZACION 

import numpy as np
from sklearn.linear_model import LassoCV

# Crear una lista de alphas entre 0.01 y 20 
alphas = np.linspace(0.01, 20, 100)  # 100 valores uniformemente distribuidos

# LassoCV con validación cruzada
model_lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
model_lasso_cv.fit(X_train, y_train)

# Mostrar el mejor alpha encontrado
print("Mejor alpha:", model_lasso_cv.alpha_)

# Predicciones
y_pred_lasso_cv = model_lasso_cv.predict(X_test)

# Evaluación
print("Regresión Lasso con CV (alpha hasta 20):")
print("MSE:", mean_squared_error(y_test, y_pred_lasso_cv))
print("R2:", r2_score(y_test, y_pred_lasso_cv))

import matplotlib.pyplot as plt

# Graficar error de validación cruzada vs alpha
plt.figure(figsize=(8, 5))
plt.plot(model_lasso_cv.alphas_, model_lasso_cv.mse_path_.mean(axis=1), marker='o')
plt.axvline(model_lasso_cv.alpha_, color='r', linestyle='--', label=f'Mejor alpha: {model_lasso_cv.alpha_:.2f}')
plt.xlabel('Alpha')
plt.ylabel('Error medio de validación cruzada (MSE)')
plt.title('LassoCV - Error vs Alpha')
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test - y_pred_lasso_cv
plt.scatter(y_pred_lasso_cv, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Residuos vs Predicciones')
plt.show()

#MODELO DE REGRESION LINEAL REGULARIZADO (RidgeCV)

from sklearn.linear_model import RidgeCV
import numpy as np

# Crear una lista de alphas entre 0.01 y 20
alphas = np.linspace(0.01, 20, 100)

# RidgeCV con validación cruzada
model_ridge_cv = RidgeCV(alphas=alphas, cv=5)
model_ridge_cv.fit(X_train, y_train)

# Mostrar el mejor alpha encontrado
print("Mejor alpha (Ridge):", model_ridge_cv.alpha_)

# Predicciones
y_pred_ridge_cv = model_ridge_cv.predict(X_test)

# Evaluación
print("Regresión Ridge con CV:")
print("MSE:", mean_squared_error(y_test, y_pred_ridge_cv))
print("R2:", r2_score(y_test, y_pred_ridge_cv))

