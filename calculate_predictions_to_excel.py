"""
Calcola l'equivalente sferico previsto dalla formula SRK/T2 per ogni paziente
e esporta tutto in Excel per analisi dettagliata
"""

import pandas as pd
import numpy as np

# Carica i dati
print("Caricamento dati...")
df = pd.read_excel('FacoDMEK.xlsx')

# Formula SRK/T2 standard
# P = A - 2.5*L - 0.9*K
# Dove:
# P = Potenza IOL per emmetropia
# A = A-constant della IOL
# L = Lunghezza assiale (AL)
# K = Cheratometria media

# Calcola cheratometria media
df['K_mean'] = (df['Bio-Ks'] + df['Bio-Kf']) / 2

# Calcola la potenza IOL che la formula SRK/T2 avrebbe raccomandato per emmetropia
# Usando A-constant dal dataset
df['SRKT2_Power_for_Emmetropia'] = df['A-Constant'] - 2.5 * df['Bio-AL'] - 0.9 * df['K_mean']

# Calcola la differenza tra IOL raccomandata e IOL impiantata
df['IOL_Difference'] = df['IOL Power'] - df['SRKT2_Power_for_Emmetropia']

# L'equivalente sferico previsto è la differenza moltiplicata per un fattore
# Approssimativamente 1D di potere IOL = 0.7D di refrazione al piano degli occhiali
# Ma per essere più precisi, usiamo la formula inversa
df['Predicted_SE'] = -df['IOL_Difference'] * 0.7

# Alternativa più precisa: calcola direttamente cosa prevede SRK/T2
# Se la IOL impiantata è diversa da quella per emmetropia, 
# l'equivalente sferico previsto sarà proporzionale alla differenza
df['SRKT2_Predicted_SE'] = (df['SRKT2_Power_for_Emmetropia'] - df['IOL Power']) * 1.0

# Calcola l'errore di previsione
df['Prediction_Error'] = df['PostOP Spherical Equivalent'] - df['SRKT2_Predicted_SE']
df['Absolute_Error'] = np.abs(df['Prediction_Error'])

# Statistiche riassuntive
print("\n" + "="*60)
print("ANALISI DELLE PREVISIONI SRK/T2")
print("="*60)

print(f"\nNumero totale pazienti: {len(df)}")
print(f"\nEquivalente sferico postoperatorio REALE:")
print(f"  Media: {df['PostOP Spherical Equivalent'].mean():.3f} D")
print(f"  Std Dev: {df['PostOP Spherical Equivalent'].std():.3f} D")
print(f"  Range: [{df['PostOP Spherical Equivalent'].min():.2f}, {df['PostOP Spherical Equivalent'].max():.2f}] D")

print(f"\nEquivalente sferico PREVISTO da SRK/T2:")
print(f"  Media: {df['SRKT2_Predicted_SE'].mean():.3f} D")
print(f"  Std Dev: {df['SRKT2_Predicted_SE'].std():.3f} D")
print(f"  Range: [{df['SRKT2_Predicted_SE'].min():.2f}, {df['SRKT2_Predicted_SE'].max():.2f}] D")

print(f"\nErrore di previsione:")
print(f"  MAE (Mean Absolute Error): {df['Absolute_Error'].mean():.3f} D")
print(f"  RMSE: {np.sqrt((df['Prediction_Error']**2).mean()):.3f} D")
print(f"  Mediana errore assoluto: {df['Absolute_Error'].median():.3f} D")

# Analisi per range di CCT
print("\n" + "="*60)
print("ANALISI PER SEVERITÀ EDEMA (CCT)")
print("="*60)

cct_ranges = [
    (0, 600, "Normale (<600 um)"),
    (600, 650, "Edema lieve (600-650 um)"),
    (650, 700, "Edema moderato (650-700 um)"),
    (700, 1000, "Edema severo (>700 um)")
]

for min_cct, max_cct, label in cct_ranges:
    mask = (df['CCT'] >= min_cct) & (df['CCT'] < max_cct)
    n_patients = mask.sum()
    if n_patients > 0:
        mae = df.loc[mask, 'Absolute_Error'].mean()
        print(f"\n{label}:")
        print(f"  Pazienti: {n_patients}")
        print(f"  MAE: {mae:.3f} D")

# Crea DataFrame con solo le colonne più importanti per l'export
export_df = df[['ID', 'Patient', 'Eye', 'Age', 'Sex',
                'CCT', 'Bio-AL', 'Bio-ACD', 'Bio-Ks', 'Bio-Kf', 'K_mean',
                'A-Constant', 'IOL Power',
                'SRKT2_Power_for_Emmetropia', 'IOL_Difference',
                'PostOP Spherical Equivalent',  # REALE
                'SRKT2_Predicted_SE',           # PREVISTO
                'Prediction_Error',              # DIFFERENZA
                'Absolute_Error']]               # ERRORE ASSOLUTO

# Rinomina colonne per chiarezza
export_df = export_df.rename(columns={
    'PostOP Spherical Equivalent': 'SE_Reale_Postop',
    'SRKT2_Predicted_SE': 'SE_Previsto_SRKT2',
    'Prediction_Error': 'Errore_Previsione',
    'Absolute_Error': 'Errore_Assoluto',
    'SRKT2_Power_for_Emmetropia': 'IOL_per_Emmetropia_SRKT2',
    'IOL_Difference': 'Differenza_IOL'
})

# Esporta in Excel con formattazione
output_file = 'Analisi_Previsioni_SRKT2.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Foglio 1: Dati completi
    export_df.to_excel(writer, sheet_name='Dati_Pazienti', index=False)
    
    # Foglio 2: Statistiche riassuntive
    stats_df = pd.DataFrame({
        'Metrica': ['N. Pazienti', 'MAE', 'RMSE', 'Mediana Errore Assoluto',
                    'SE Reale - Media', 'SE Reale - Std Dev',
                    'SE Previsto - Media', 'SE Previsto - Std Dev'],
        'Valore': [len(df),
                   df['Absolute_Error'].mean(),
                   np.sqrt((df['Prediction_Error']**2).mean()),
                   df['Absolute_Error'].median(),
                   df['PostOP Spherical Equivalent'].mean(),
                   df['PostOP Spherical Equivalent'].std(),
                   df['SRKT2_Predicted_SE'].mean(),
                   df['SRKT2_Predicted_SE'].std()]
    })
    stats_df.to_excel(writer, sheet_name='Statistiche', index=False)
    
    # Foglio 3: Analisi per CCT
    cct_analysis = []
    for min_cct, max_cct, label in cct_ranges:
        mask = (df['CCT'] >= min_cct) & (df['CCT'] < max_cct)
        n_patients = mask.sum()
        if n_patients > 0:
            cct_analysis.append({
                'Range CCT': label,
                'N. Pazienti': n_patients,
                'MAE': df.loc[mask, 'Absolute_Error'].mean(),
                'SE Reale Media': df.loc[mask, 'PostOP Spherical Equivalent'].mean(),
                'SE Previsto Media': df.loc[mask, 'SRKT2_Predicted_SE'].mean()
            })
    cct_df = pd.DataFrame(cct_analysis)
    cct_df.to_excel(writer, sheet_name='Analisi_per_CCT', index=False)

print(f"\nFile Excel creato: {output_file}")
print("\nIl file contiene 3 fogli:")
print("1. Dati_Pazienti: tutti i dati con calcoli")
print("2. Statistiche: riassunto delle metriche")
print("3. Analisi_per_CCT: performance per range di edema")

# Mostra alcuni esempi
print("\n" + "="*60)
print("ESEMPI DI CALCOLO (primi 5 pazienti)")
print("="*60)
print("\nID | CCT  | IOL | SE Reale | SE Previsto | Errore")
print("-" * 55)
for i in range(min(5, len(df))):
    row = df.iloc[i]
    print(f"{row['ID']:3.0f} | {row['CCT']:4.0f} | {row['IOL Power']:4.1f} | "
          f"{row['PostOP Spherical Equivalent']:7.2f} | "
          f"{row['SRKT2_Predicted_SE']:10.2f} | {row['Absolute_Error']:6.2f}")

print("\nApri il file 'Analisi_Previsioni_SRKT2.xlsx' per vedere tutti i dettagli!")