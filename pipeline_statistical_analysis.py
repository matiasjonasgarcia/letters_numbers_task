import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

#--------------------------------------------------------------------------------#
# 0. Carga de datos
#--------------------------------------------------------------------------------#
df = pd.read_excel('Base_LN_limpia.xlsx')

# Definición de variables
vars_descriptive = {
    'FC-KLNPREC_TOTAL': 'Total Precision',
    'FC-KLNPREC_RR': 'Precision repeated responses',
    'FC-KLNPREC_RN': 'Precision novel responses',
    'FC-KLNTR_TOTAL': 'Total RT',
    'FC-KLNTR_RR': 'RT repeated responses',
    'FC-KLNTR_RN': 'RT novel responses',
    'FC-KLNVSR': 'Voluntary Switch Rate',
    'FC-KLNCRT': 'Selection Response Time',
    'FC-KLN_PREC1': 'Precision in instance 1',
    'FC-KLN_PREC2': 'Precision in instance 2',
    'FC-KLN_LVL1': 'RT-instance 1',
    'FC-KLN_LVL2': 'RT-instance 2',
    'FC-KLN_LVL3': 'RT-instance 3',
    'FC-KLN_LVL4': 'RT-instance 4',
    'FC-KLNPERC_NUM': '% Number responses',
    'FC-KLNPERC_LETRA': '% Letter responses',
    'FC-KLNTR_N1': 'RT N-1',
    'FC-KLNTR_S': 'RT N',
    'FC-KLNTR_P1': 'RT P1'
}

#--------------------------------------------------------------------------------#
# --- BLOQUE 1: DESCRIPTIVE ANALYSIS ---
#--------------------------------------------------------------------------------#
print("--- 1. DESCRIPTIVE ANALYSIS ---")
desc_stats = df[list(vars_descriptive.keys())].agg(['mean', 'std', 'skew', 'kurtosis']).T
desc_stats.index = desc_stats.index.map(vars_descriptive)
print(desc_stats)
desc_stats.to_excel('descriptive_analysis.xlsx')

#--------------------------------------------------------------------------------#
# --- BLOQUE 2: INFERENTIAL ANALYSIS ---
#--------------------------------------------------------------------------------#
print("\n--- 2. INFERENTIAL ANALYSIS ---")

# a) Repetition bias (Wilcoxon)
res_a = pg.wilcoxon(df['FC-KLNPERC_LETRA'], df['FC-KLNPERC_NUM'])

# DEBUG: Descomenta la siguiente línea si quieres ver exactamente qué columnas tiene tu versión
#print("Columnas detectadas en Wilcoxon:", res_a.columns.tolist())

# Acceso seguro por posición: Generalmente el estadístico es la col 0 y el p-val es la col 1 o 2
# Pero para ser 100% precisos, extraemos por nombre si existe, o por posición si no.
#stat_col = 'W-val' if 'W-val' in res_a.columns else ('W' if 'W' in res_a.columns else res_a.columns[0])
#p_col = 'p-val'

print(f"a) Repetition bias: W={res_a['W_val'].values[0]:.3f}, p={res_a['p_val'].values[0]:.3f}")

# b) Performance bias (Wilcoxon)
res_b_prec = pg.wilcoxon(df['FC-KLNPREC_RR'], df['FC-KLNPREC_RN'])
res_b_rt = pg.wilcoxon(df['FC-KLNTR_RR'], df['FC-KLNTR_RN'])

print(f"b) Performance bias (Precision): W={res_b_prec['W_val'].values[0]:.3f}, p={res_b_prec['p_val'].values[0]:.3f}")
print(f"   Performance bias (RT): W={res_b_rt['W_val'].values[0]:.3f}, p={res_b_rt['p_val'].values[0]:.3f}")

# c) ANOVA RM: RT N-1, N, P1 con Greenhouse-Geisser
# Pasamos a formato largo (long format)
df_long_c = pd.melt(df.reset_index(), id_vars=['index'], 
                    value_vars=['FC-KLNTR_N1', 'FC-KLNTR_S', 'FC-KLNTR_P1'],
                    var_name='Position', value_name='RT')

anova_c = pg.rm_anova(data=df_long_c, dv='RT', within='Position', subject='index', correction=True)

# Esto te dirá explícitamente si se cumple la esfericidad
print(pg.sphericity(data=df_long_c, dv='RT', within='Position', subject='index'))

print("\nc) ANOVA RM (RT N-1, N, P1) con corrección GG:")
# Selecciona solo las columnas que están presentes en el resultado
cols_c = [c for c in ['Source', 'ddof1', 'ddof2', 'F', 'p_unc', 'p_GG', 'eps', 'ng2'] if c in anova_c.columns]
print(anova_c[cols_c])

# ---Post-hoc para ANOVA RM (Position) ---
posthoc_c = pg.pairwise_tests(data=df_long_c, dv='RT', within='Position', 
                              subject='index', padjust='bonferroni')
print("\nPost-hoc (Bonferroni) - Position:")
# Filtramos las columnas clave para el reporte
cols_post = [c for c in ['A', 'B', 'T', 'dof', 'p_unc', 'p_corr', 'hedges'] if c in posthoc_c.columns]
print(posthoc_c[cols_post])


# d) ANOVA RM: RT-instances 1-4 con Greenhouse-Geisser
df_long_d = pd.melt(df.reset_index(), id_vars=['index'], 
                    value_vars=['FC-KLN_LVL1', 'FC-KLN_LVL2', 'FC-KLN_LVL3', 'FC-KLN_LVL4'],
                    var_name='Instance', value_name='RT')

anova_d = pg.rm_anova(data=df_long_d, dv='RT', within='Instance', subject='index', correction=True)
print("\nd) ANOVA RM (RT-instances 1-4) con corrección GG:")
cols_d = [c for c in ['Source', 'ddof1', 'ddof2', 'F', 'p_unc', 'p_GG', 'eps', 'ng2'] if c in anova_d.columns]
print(anova_d[cols_d])

# ---Post-hoc para ANOVA RM (Instances) ---
# Solo es necesario si el ANOVA del bloque D también fue significativo (p < .05)
posthoc_d = pg.pairwise_tests(data=df_long_d, dv='RT', within='Instance', 
                              subject='index', padjust='bonferroni')
print("\nPost-hoc (Bonferroni) - Instances:")
print(posthoc_d[cols_post])


print("\n--- 3. ANÁLISIS DE POSICIÓN (LMM Y NO PARAMÉTRICO) ---")

# 1. Limpiar nulos y resetear índice
df_long_c_clean = df_long_c.dropna(subset=['RT', 'Position', 'index']).reset_index(drop=True)

# 2. Modelo Lineal Mixto (LMM)
modelo_posicion = smf.mixedlm(
    "RT ~ C(Position)", 
    data=df_long_c_clean,  # Corregido: ahora usa la base limpia
    groups=df_long_c_clean["index"]
)
resultados_posicion = modelo_posicion.fit()
print("\na) Resumen del LMM - Posición (N-1, N, P1):")
print(resultados_posicion.summary())

# 3. Prueba de Friedman (Efecto principal no paramétrico)
# Nota: Friedman requiere datos balanceados. Si algún sujeto tiene datos incompletos, 
# Pingouin avisará. En tareas computarizadas suele estar balanceado.
try:
    friedman_pos = pg.friedman(data=df_long_c_clean, dv='RT', within='Position', subject='index')
    print("\nb) Prueba de Friedman - Posición:")
    print(friedman_pos)
except Exception as e:
    print(f"\nb) No se pudo correr Friedman (los datos no están perfectamente balanceados): {e}")

# 4. Post-Hoc de Wilcoxon con corrección de Bonferroni
posthoc_pos = pg.pairwise_tests(
    data=df_long_c_clean, 
    dv='RT', 
    within='Position', 
    subject='index', 
    parametric=False, # Esto transforma la prueba en Wilcoxon
    padjust='bonf'  # Corrección para comparaciones múltiples
)
print("\nc) Post-Hocs No Paramétricos (Wilcoxon) - Posición:")
print(posthoc_pos[['A', 'B', 'W_val', 'p_unc', 'p_corr', 'hedges']])


print("\n" + "="*50 + "\n")



# b) LMM para Instancias (1 a 4)
# 1. Limpiar nulos y resetear índice
df_long_d_clean = df_long_d.dropna(subset=['RT', 'Instance', 'index']).reset_index(drop=True)

# 2. Modelo Lineal Mixto (LMM)
modelo_instancia = smf.mixedlm(
    "RT ~ C(Instance)", 
    data=df_long_d_clean, 
    groups=df_long_d_clean["index"]
)
resultados_instancia = modelo_instancia.fit()
print("\na) Resumen del LMM - Instancias (1 a 4):")
print(resultados_instancia.summary())

# 3. Prueba de Friedman (Efecto principal no paramétrico)
try:
    friedman_inst = pg.friedman(data=df_long_d_clean, dv='RT', within='Instance', subject='index')
    print("\nb) Prueba de Friedman - Instancias:")
    print(friedman_inst)
except Exception as e:
    print(f"\nb) No se pudo correr Friedman (los datos no están perfectamente balanceados): {e}")

# 4. Post-Hoc de Wilcoxon con corrección de Bonferroni
posthoc_inst = pg.pairwise_tests(
    data=df_long_d_clean, 
    dv='RT', 
    within='Instance', 
    subject='index', 
    parametric=False, # Transformado a Wilcoxon
    padjust='bonf'
)
print("\nc) Post-Hocs No Paramétricos (Wilcoxon) - Instancias:")
print(posthoc_inst[['A', 'B', 'W_val', 'p_unc', 'p_corr', 'hedges']])



# e) Pearson Correlations
vars_corr = [
    'FC-KLNPREC_RR', 'FC-KLNPREC_RN', 'FC-KLNTR_RR', 'FC-KLNTR_RN', 'FC-KLNCRT',
    'FC_DedosECCP', 'FC_DedosECCTR', 'FC_DedosESCP', 'FC_DedosESCTR'
]
corr_matrix = df[vars_corr].rcorr(method='pearson', stars=True)
print("\ne) Matriz de Correlación (r arriba, p-valor abajo):")
print(corr_matrix)

#--------------------------------------------------------------------------------#
# --- BLOQUE 3: FIGURES ---
#--------------------------------------------------------------------------------#
# Usamos pointplot para mostrar mejor la tendencia de medidas repetidas
sns.set_theme(style="whitegrid")


# --- Figura A: ANOVA N-1, N, P1 ---
plt.figure(figsize=(8, 6))  # Crea una figura nueva
sns.pointplot(data=df_long_c, x='Position', y='RT', capsize=.1, errorbar='se', color='blue')
plt.title('Reaction Time by Position (N-1, N, P1)')
plt.ylabel('Mean RT (ms)')
plt.xlabel('Trial Position')
plt.xticks(ticks=[0, 1, 2], labels=['N-1', 'N', 'P1'])
plt.tight_layout()
plt.savefig('figure_A_position_RT.png', dpi=300) # Alta resolución para el artículo
plt.show() # Muestra esta figura sola

# --- Figura B: ANOVA Instances 1-4 ---
plt.figure(figsize=(8, 6))  # Crea otra figura nueva independiente
sns.pointplot(data=df_long_d, x='Instance', y='RT', capsize=.1, errorbar='se', color='darkred')
plt.title('Reaction Time by Instance (1-4)')
plt.ylabel('Mean RT (ms)')
plt.xlabel('Task Instance')
plt.xticks(ticks=[0, 1, 2, 3], labels=['I1', 'I2', 'I3', 'I4'])
plt.tight_layout()
plt.savefig('figure_B_instances_RT.png', dpi=300) # Alta resolución
plt.show() # Muestra esta segunda figura

print("\n--- 3. ANÁLISIS SOCIODEMOGRÁFICO ---")

# 1. Verificamos que la variable exista en la base
if 'ESCOLARIDAD' in df.columns:
    # Diccionario para traducir los números de vuelta a texto para el gráfico
    mapa_escolaridad_inverso = {
        1.0: 'Primario', 
        2.0: 'Secundario', 
        3.0: 'Secundario Completo', 
        4.0: 'Universitario Incompleto', 
        5.0: 'Universitario/Terciario Completo', 
        99.0: 'Otro / Sin Datos'
    }
    
    # Asegurarnos de que sea numérica para el mapeo
    df['ESCOLARIDAD_LABEL'] = pd.to_numeric(df['ESCOLARIDAD'], errors='coerce').map(mapa_escolaridad_inverso)
    
    # Calcular porcentajes
    porcentajes = df['ESCOLARIDAD_LABEL'].value_counts(normalize=True) * 100
    
    # Orden lógico de menor a mayor escolaridad para el gráfico
    orden_logico = [
        'Primario', 'Secundario', 'Secundario Completo', 
        'Universitario Incompleto', 'Universitario/Terciario Completo', 'Otro / Sin Datos'
    ]
    # Filtramos solo las categorías que realmente tienen datos
    porcentajes = porcentajes.reindex([cat for cat in orden_logico if cat in porcentajes.index])
    
    print("Porcentajes de Escolaridad:")
    print(porcentajes.round(2).astype(str) + " %")
    
    # 2. Crear el Gráfico de Barras Horizontal
    plt.figure(figsize=(10, 6))
    
    # Usamos seaborn para darle un estilo moderno y estético
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=porcentajes.values, y=porcentajes.index, palette="viridis")
    
    # Títulos y etiquetas
    plt.title('Distribución del Nivel de Escolaridad', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Porcentaje de la Muestra (%)', fontsize=12)
    plt.ylabel('Nivel Educativo', fontsize=12)
    
    # Agregar las etiquetas de texto (los números exactos) al final de cada barra
    for i, v in enumerate(porcentajes.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", color='black', va='center', fontweight='bold')
        
    # Ajustar el diseño para que no se corten los textos largos
    plt.tight_layout()
    
    # Guardar la imagen en alta calidad (300 dpi es el estándar para revistas científicas)
    nombre_archivo = 'grafico_escolaridad.png'
    plt.savefig(nombre_archivo, dpi=300)
    print(f"\n¡Gráfico generado exitosamente! Se ha guardado como '{nombre_archivo}' en tu carpeta.")
    
else:
    print("La variable 'ESCOLARIDAD' no se encontró en la base de datos.")

# =============================================================================
# PREPARACIÓN DE DATOS (Predicciones del Modelo)
# =============================================================================
# Obtenemos los valores predichos por el modelo (Efectos Fijos) para trazar 
# la tendencia general "pura", libre del ruido aleatorio de cada sujeto.
df_long_c['Predicted_RT'] = resultados_posicion.predict(df_long_c_clean)
df_long_d['Predicted_RT'] = resultados_instancia.predict(df_long_d_clean)

sns.set_theme(style="whitegrid")

# =============================================================================
# ANÁLISIS 1: POSICIÓN (N-1, N, P1)
# =============================================================================

# 1A. SPAGHETTI PLOT - Posición
plt.figure(figsize=(10, 6))
# 1. Trazamos las líneas individuales por sujeto (finas y semitransparentes)
sns.lineplot(data=df_long_c, x='Position', y='RT', units='index', estimator=None, 
             color='grey', alpha=0.3, linewidth=1)
# 2. Superponemos la línea de medias predichas por el modelo (efecto fijo general)
sns.lineplot(data=df_long_c, x='Position', y='Predicted_RT', 
             color='red', linewidth=3.5, marker='o', markersize=8, errorbar=None, 
             label='Media Predicha (LMM)')

plt.title('Spaghetti Plot: Variabilidad Individual vs. Modelo (Posición)', fontsize=14, fontweight='bold')
plt.ylabel('Reaction Time (ms)', fontsize=12)
plt.xlabel('Trial Position', fontsize=12)
plt.xticks(ticks=[0, 1, 2], labels=['N-1', 'N', 'P1'])
plt.legend()
plt.tight_layout()
plt.savefig('figure_C1_Spaghetti_Position.png', dpi=300)
plt.show()

# 1B. EMM PLOT (Medias Marginales Estimadas) - Posición
plt.figure(figsize=(8, 6))
# Trazamos la media observada empírica con su Error Estándar (SE)
sns.pointplot(data=df_long_c, x='Position', y='RT', errorbar='se', 
              color='lightblue', label='Media Observada (± SE)', markers='s', alpha=0.7)
# Trazamos la media predicha por el LMM (EMM)
sns.pointplot(data=df_long_c, x='Position', y='Predicted_RT', errorbar=None, 
              color='darkblue', linestyles='--', label='Media Predicha LMM (EMM)', markers='D')

plt.title('Medias Marginales Estimadas vs. Observadas (Posición)', fontsize=14, fontweight='bold')
plt.ylabel('Mean RT (ms)', fontsize=12)
plt.xlabel('Trial Position', fontsize=12)
plt.xticks(ticks=[0, 1, 2], labels=['N-1', 'N', 'P1'])
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figure_C2_EMM_Position.png', dpi=300)
plt.show()

# =============================================================================
# ANÁLISIS 2: INSTANCIAS (1 al 4)
# =============================================================================

# 2A. SPAGHETTI PLOT - Instancias
plt.figure(figsize=(10, 6))
# Trazamos las líneas individuales
sns.lineplot(data=df_long_d, x='Instance', y='RT', units='index', estimator=None, 
             color='grey', alpha=0.3, linewidth=1)
# Superponemos la línea predicha del modelo
sns.lineplot(data=df_long_d, x='Instance', y='Predicted_RT', 
             color='darkred', linewidth=3.5, marker='o', markersize=8, errorbar=None, 
             label='Media Predicha (LMM)')

plt.title('Spaghetti Plot: Variabilidad Individual vs. Modelo (Instancias)', fontsize=14, fontweight='bold')
plt.ylabel('Reaction Time (ms)', fontsize=12)
plt.xlabel('Task Instance', fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3], labels=['I1', 'I2', 'I3', 'I4'])
plt.legend()
plt.tight_layout()
plt.savefig('figure_D1_Spaghetti_Instances.png', dpi=300)
plt.show()

# 2B. EMM PLOT (Medias Marginales Estimadas) - Instancias
plt.figure(figsize=(8, 6))
# Observadas (empíricas)
sns.pointplot(data=df_long_d, x='Instance', y='RT', errorbar='se', 
              color='salmon', label='Media Observada (± SE)', markers='s', alpha=0.7)
# Predichas (EMM)
sns.pointplot(data=df_long_d, x='Instance', y='Predicted_RT', errorbar=None, 
              color='darkred', linestyles='--', label='Media Predicha LMM (EMM)', markers='D')

plt.title('Medias Marginales Estimadas vs. Observadas (Instancias)', fontsize=14, fontweight='bold')
plt.ylabel('Mean RT (ms)', fontsize=12)
plt.xlabel('Task Instance', fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3], labels=['I1', 'I2', 'I3', 'I4'])
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figure_D2_EMM_Instances.png', dpi=300)
plt.show()