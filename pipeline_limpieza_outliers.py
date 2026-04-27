import pandas as pd
import numpy as np

# 1. Cargar la base de datos (la que ya tiene los decimales corregidos)
df = pd.read_excel('Base_LN_bruta03.xlsx')

# 2. Definir en qué columnas buscaremos los valores extremos
# Usaremos las variables principales de tiempo de reacción de Letras y Números
columnas_rt = [
    'FC-KLN_LVL1', 'FC-KLN_LVL2', 'FC-KLN_LVL3', 'FC-KLN_LVL4',
    'FC-KLNTR_TOTAL', 'FC-KLNTR_RR', 'FC-KLNTR_RN', 'FC-KLN_TR1', 'FC-KLN_TR2'
]

# Validar cuáles de estas columnas realmente existen en tu base
columnas_rt = [col for col in columnas_rt if col in df.columns]

# 3. Función para limpiar outliers a nivel grupal (+/- 2.5 SD)
def limpiar_outliers_grupales(dataframe, columnas, umbral_sd=2.5):
    df_limpio = dataframe.copy()
    
    print("--- REPORTE DE LIMPIEZA DE OUTLIERS ---")
    
    for col in columnas:
        # Calcular la Media y la Desviación Estándar de toda la muestra para esa columna
        media = df_limpio[col].mean()
        std = df_limpio[col].std()
        
        # Definir los límites superior e inferior
        limite_inferior = media - (umbral_sd * std)
        limite_superior = media + (umbral_sd * std)
        
        # Identificar cuántos casos caen fuera del límite
        outliers = (df_limpio[col] < limite_inferior) | (df_limpio[col] > limite_superior)
        cantidad_outliers = outliers.sum()
        
        if cantidad_outliers > 0:
            print(f"Columna '{col}': Se detectaron {cantidad_outliers} valores atípicos.")
            print(f"  -> Rango aceptado: {limite_inferior:.2f} ms a {limite_superior:.2f} ms")
            
            # Reemplazar los valores atípicos por NaN (Nulo)
            # Esto evita que el outlier arruine el promedio, sin borrar al sujeto de la base
            df_limpio.loc[outliers, col] = np.nan
        else:
            print(f"Columna '{col}': 0 valores atípicos detectados.")
            
    print("---------------------------------------")
    return df_limpio

# 4. Aplicar la limpieza
df_sin_outliers = limpiar_outliers_grupales(df, columnas_rt, umbral_sd=2.5)

# 5. Exportar la base de datos lista para estadística inferencial
df_sin_outliers.to_excel('Base_LN_limpia.xlsx', index=False)
print("\n¡Base lista! Exportada como 'Base_LN_limpia.xlsx'.")