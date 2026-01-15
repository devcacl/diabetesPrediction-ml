"""
Sistema de Predição de Diabetes usando XGBoost
===============================================

PROBLEMA REAL: 
Diabetes afeta milhões de pessoas. Detectar precocemente pode salvar vidas.
Este sistema usa Machine Learning para prever diabetes baseado em dados médicos.

ALGORITMO: XGBoost (Extreme Gradient Boosting)
- Usa múltiplos árvores de decisão
- Cada árvore aprende com os erros da anterior
- Excelente para dados médicos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("🏥 SISTEMA DE PREDIÇÃO DE DIABETES - TREINAMENTO DO MODELO")
print("="*70)

# ============================================
# PASSO 1: CARREGAR DADOS
# ============================================
print("\n📂 PASSO 1: Carregando dados do diabetes...")

# Carregar o dataset
df = pd.read_csv('../data/diabetes.csv')

print(f"✅ Dados carregados com sucesso!")
print(f"📊 Total de pacientes: {len(df)}")
print(f"📋 Colunas disponíveis: {list(df.columns)}")
print(f"\n🔍 Primeiras 5 linhas:")
print(df.head())

# Verificar valores faltantes
print(f"\n❓ Valores faltantes por coluna:")
print(df.isnull().sum())

# Distribuição de classes
print(f"\n📊 Distribuição de Diabetes:")
print(f"   Sem diabetes (0): {len(df[df['Outcome']==0])} pacientes")
print(f"   Com diabetes (1): {len(df[df['Outcome']==1])} pacientes")
print(f"   Proporção: {(df['Outcome'].sum()/len(df)*100):.1f}% tem diabetes")

# ============================================
# PASSO 2: ANÁLISE EXPLORATÓRIA
# ============================================
print("\n" + "="*70)
print("📈 PASSO 2: Análise exploratória dos dados")
print("="*70)

# Estatísticas básicas
print("\n📊 Estatísticas descritivas:")
print(df.describe())

# Verificar correlação com diabetes
print("\n🔗 Correlação com Diabetes:")
correlations = df.corr()['Outcome'].sort_values(ascending=False)
print(correlations)

# ============================================
# PASSO 3: PREPARAR DADOS
# ============================================
print("\n" + "="*70)
print("🔧 PASSO 3: Preparando dados para o modelo")
print("="*70)

# Separar features (X) e target (y)
# X = todas as colunas EXCETO Outcome
# y = apenas a coluna Outcome (0 ou 1)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"✅ Features (X): {X.shape[1]} variáveis")
print(f"   {list(X.columns)}")
print(f"✅ Target (y): Outcome (0=Não diabético, 1=Diabético)")

# ============================================
# PASSO 4: DIVIDIR EM TREINO E TESTE
# ============================================
print("\n" + "="*70)
print("✂️ PASSO 4: Dividindo dados em treino (80%) e teste (20%)")
print("="*70)

# train_test_split divide os dados
# test_size=0.2 significa 20% para teste
# random_state=42 garante que sempre dividimos da mesma forma
# stratify=y mantém a mesma proporção de diabetes em ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para teste
    random_state=42,     # Reprodutibilidade
    stratify=y           # Manter proporção de classes
)

print(f"📚 Conjunto de TREINO: {len(X_train)} pacientes")
print(f"   - Sem diabetes: {len(y_train[y_train==0])}")
print(f"   - Com diabetes: {len(y_train[y_train==1])}")
print(f"\n🧪 Conjunto de TESTE: {len(X_test)} pacientes")
print(f"   - Sem diabetes: {len(y_test[y_test==0])}")
print(f"   - Com diabetes: {len(y_test[y_test==1])}")

# ============================================
# PASSO 5: NORMALIZAR DADOS
# ============================================
print("\n" + "="*70)
print("⚖️ PASSO 5: Normalizando dados")
print("="*70)

print("❓ Por que normalizar?")
print("   - Glicose pode variar de 0-200")
print("   - Idade pode variar de 20-80")
print("   - Normalizar coloca tudo na mesma escala")
print("   - Isso ajuda o modelo a aprender melhor!")

# StandardScaler transforma os dados para média=0 e desvio=1
scaler = StandardScaler()

# fit_transform: aprende a escala do treino e transforma
X_train_scaled = scaler.fit_transform(X_train)

# transform: usa a mesma escala do treino
X_test_scaled = scaler.transform(X_test)

print("✅ Dados normalizados!")

# Salvar o scaler para usar na API depois
joblib.dump(scaler, '../models/scaler.pkl')
print("💾 Scaler salvo em: ../models/scaler.pkl")

# ============================================
# PASSO 6: TREINAR MODELO XGBOOST
# ============================================
print("\n" + "="*70)
print("🧠 PASSO 6: Treinando modelo XGBoost")
print("="*70)

print("\n📖 Como funciona o XGBoost?")
print("1. Cria primeira árvore de decisão")
print("2. Vê onde a árvore errou")
print("3. Cria nova árvore para corrigir os erros")
print("4. Repete isso 200 vezes (n_estimators=200)")
print("5. Combina todas as árvores para fazer a predição final")

# Criar e configurar o modelo
model = xgb.XGBClassifier(
    max_depth=5,              # Profundidade máxima de cada árvore
    learning_rate=0.1,        # Velocidade de aprendizado
    n_estimators=200,         # Número de árvores
    objective='binary:logistic',  # Classificação binária (0 ou 1)
    eval_metric='logloss',    # Métrica de avaliação
    random_state=42,          # Reprodutibilidade
    use_label_encoder=False   # Evitar warnings
)

print("\n🎯 Iniciando treinamento...")
# Treinar o modelo
model.fit(X_train_scaled, y_train)

print("✅ Treinamento concluído!")

# ============================================
# PASSO 7: AVALIAR O MODELO
# ============================================
print("\n" + "="*70)
print("📊 PASSO 7: Avaliando desempenho do modelo")
print("="*70)

# Fazer predições no conjunto de teste
y_pred = model.predict(X_test_scaled)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📈 MÉTRICAS DO MODELO:")
print("="*70)
print(f"🎯 ACURÁCIA (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
print("   → De todas as predições, quantas estão corretas?")
print(f"\n🎯 PRECISÃO (Precision): {precision:.4f} ({precision*100:.2f}%)")
print("   → Quando o modelo diz 'tem diabetes', quantas vezes está certo?")
print(f"\n🎯 RECALL (Sensibilidade): {recall:.4f} ({recall*100:.2f}%)")
print("   → De todos que TÊM diabetes, quantos o modelo detectou?")
print(f"\n🎯 F1-SCORE:              {f1:.4f} ({f1*100:.2f}%)")
print("   → Média harmônica entre precisão e recall")
print("="*70)

# Relatório de classificação detalhado
print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred, 
                          target_names=['Sem Diabetes', 'Com Diabetes']))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\n🎲 MATRIZ DE CONFUSÃO:")
print("="*70)
print(cm)
print("\nComo ler:")
print(f"   ✅ Verdadeiros Negativos (TN): {cm[0][0]} - Sem diabetes, predito corretamente")
print(f"   ❌ Falsos Positivos (FP):  {cm[0][1]} - Sem diabetes, mas predito como diabético")
print(f"   ❌ Falsos Negativos (FN):  {cm[1][0]} - Com diabetes, mas predito como não diabético")
print(f"   ✅ Verdadeiros Positivos (TP): {cm[1][1]} - Com diabetes, predito corretamente")

# Visualizar matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sem Diabetes', 'Com Diabetes'],
            yticklabels=['Sem Diabetes', 'Com Diabetes'])
plt.title('Matriz de Confusão - Predição de Diabetes', fontsize=16, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Predição', fontsize=12)
plt.tight_layout()
plt.savefig('../models/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n💾 Matriz de confusão salva em: ../models/confusion_matrix.png")

# Importância das características
print("\n" + "="*70)
print("🔍 IMPORTÂNCIA DAS CARACTERÍSTICAS")
print("="*70)
print("Quais variáveis o modelo considera mais importantes?")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualizar importância
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importância', fontsize=12)
plt.title('Importância das Características para Predição de Diabetes', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../models/feature_importance.png', dpi=300, bbox_inches='tight')
print("\n💾 Gráfico salvo em: ../models/feature_importance.png")

# ============================================
# PASSO 8: SALVAR O MODELO
# ============================================
print("\n" + "="*70)
print("💾 PASSO 8: Salvando modelo treinado")
print("="*70)

# Salvar modelo
joblib.dump(model, '../models/xgboost_model.pkl')
print("✅ Modelo salvo em: ../models/xgboost_model.pkl")

# Salvar nomes das features
joblib.dump(X.columns.tolist(), '../models/feature_names.pkl')
print("✅ Features salvas em: ../models/feature_names.pkl")

# ============================================
# TESTE FINAL
# ============================================
print("\n" + "="*70)
print("🧪 TESTE FINAL: Fazendo uma predição de exemplo")
print("="*70)

# Exemplo de um paciente
exemplo = X_test.iloc[0:1]
print("\n👤 Dados do paciente:")
print(exemplo)

# Normalizar
exemplo_scaled = scaler.transform(exemplo)

# Predizer
pred = model.predict(exemplo_scaled)[0]
prob = model.predict_proba(exemplo_scaled)[0]

print(f"\n🎯 PREDIÇÃO: {'COM DIABETES' if pred == 1 else 'SEM DIABETES'}")
print(f"📊 Probabilidade:")
print(f"   - Sem diabetes: {prob[0]*100:.2f}%")
print(f"   - Com diabetes: {prob[1]*100:.2f}%")
print(f"\n✅ Valor real: {'COM DIABETES' if y_test.iloc[0] == 1 else 'SEM DIABETES'}")

print("\n" + "="*70)
print("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*70)
print("\n✅ Próximos passos:")
print("   1. Criar API Flask (backend/app.py)")
print("   2. Criar interface React (frontend)")
print("   3. Integrar tudo!")
print("="*70)