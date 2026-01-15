"""
API Flask para Predição de Diabetes
====================================

O QUE FAZ ESTA API?
- Recebe dados de um paciente via HTTP
- Usa o modelo treinado para fazer predição
- Retorna se o paciente tem risco de diabetes

ENDPOINTS:
- GET  /              → Informações da API
- GET  /health        → Verifica se API está funcionando
- GET  /model-info    → Informações sobre o modelo
- POST /predict       → Fazer predição de diabetes
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# ============================================
# CONFIGURAÇÃO DA API
# ============================================

# Criar aplicação Flask
app = Flask(__name__)

# CORS permite que o React (frontend) se comunique com Flask (backend)
# Sem isso, o navegador bloqueia a comunicação
CORS(app)

# ============================================
# CARREGAR MODELO E FERRAMENTAS
# ============================================

print("🔄 Carregando modelo e ferramentas...")

try:
    # Carregar modelo treinado
    model = joblib.load('../models/xgboost_model.pkl')
    print("✅ Modelo carregado!")
    
    # Carregar scaler (para normalizar dados)
    scaler = joblib.load('../models/scaler.pkl')
    print("✅ Scaler carregado!")
    
    # Carregar nomes das features
    feature_names = joblib.load('../models/feature_names.pkl')
    print("✅ Features carregadas!")
    print(f"📋 Features esperadas: {feature_names}")
    
except Exception as e:
    print(f"❌ ERRO ao carregar modelo: {e}")
    print("💡 Execute primeiro: python train_model.py")
    model = None
    scaler = None
    feature_names = None

# ============================================
# ENDPOINT 1: PÁGINA INICIAL
# ============================================

@app.route('/', methods=['GET'])
def home():
    """
    Endpoint de boas-vindas
    
    Acesse: http://localhost:5000/
    Método: GET
    """
    return jsonify({
        'message': '🏥 API de Predição de Diabetes',
        'status': 'online',
        'version': '1.0',
        'algorithm': 'XGBoost',
        'description': 'Sistema que prediz risco de diabetes baseado em dados médicos',
        'endpoints': {
            'GET /': 'Informações da API',
            'GET /health': 'Status de saúde',
            'GET /model-info': 'Informações do modelo',
            'POST /predict': 'Fazer predição'
        }
    })

# ============================================
# ENDPOINT 2: VERIFICAR SAÚDE DA API
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """
    Verifica se a API está funcionando
    
    Acesse: http://localhost:5000/health
    Método: GET
    """
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

# ============================================
# ENDPOINT 3: INFORMAÇÕES DO MODELO
# ============================================

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Retorna informações sobre o modelo
    
    Acesse: http://localhost:5000/model-info
    Método: GET
    """
    if model is None:
        return jsonify({
            'error': 'Modelo não carregado. Execute train_model.py primeiro.'
        }), 500
    
    return jsonify({
        'algorithm': 'XGBoost (Extreme Gradient Boosting)',
        'description': 'Algoritmo de árvores de decisão com gradient boosting',
        'features': feature_names,
        'n_features': len(feature_names),
        'feature_descriptions': {
            'Pregnancies': 'Número de gestações',
            'Glucose': 'Nível de glicose no sangue',
            'BloodPressure': 'Pressão arterial (mm Hg)',
            'SkinThickness': 'Espessura da pele (mm)',
            'Insulin': 'Nível de insulina (mu U/ml)',
            'BMI': 'Índice de Massa Corporal',
            'DiabetesPedigreeFunction': 'Histórico familiar de diabetes',
            'Age': 'Idade (anos)'
        }
    })

# ============================================
# ENDPOINT 4: FAZER PREDIÇÃO (PRINCIPAL)
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal para predição de diabetes
    
    URL: http://localhost:5000/predict
    Método: POST
    
    ENTRADA (JSON):
    {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    
    SAÍDA (JSON):
    {
        "prediction": "Com Diabetes" ou "Sem Diabetes",
        "has_diabetes": true/false,
        "probability": 0.85,
        "confidence": "High",
        "risk_level": "Alto",
        "recommendations": [...]
    }
    """
    
    try:
        # ============================================
        # PASSO 1: RECEBER E VALIDAR DADOS
        # ============================================
        
        # Receber dados JSON do request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Nenhum dado recebido',
                'message': 'Envie um JSON com os dados do paciente'
            }), 400
        
        print(f"\n📥 Dados recebidos: {data}")
        
        # ============================================
        # PASSO 2: VALIDAR CAMPOS OBRIGATÓRIOS
        # ============================================
        
        required_fields = feature_names
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Campos faltantes',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400
        
        # ============================================
        # PASSO 3: VALIDAR VALORES
        # ============================================
        
        # Validações médicas básicas
        validations = {
            'Pregnancies': (0, 20, 'Número de gestações deve estar entre 0 e 20'),
            'Glucose': (0, 300, 'Glicose deve estar entre 0 e 300 mg/dL'),
            'BloodPressure': (0, 200, 'Pressão arterial deve estar entre 0 e 200 mm Hg'),
            'SkinThickness': (0, 100, 'Espessura da pele deve estar entre 0 e 100 mm'),
            'Insulin': (0, 900, 'Insulina deve estar entre 0 e 900 mu U/ml'),
            'BMI': (0, 70, 'IMC deve estar entre 0 e 70'),
            'DiabetesPedigreeFunction': (0, 3, 'Função de pedigree deve estar entre 0 e 3'),
            'Age': (1, 120, 'Idade deve estar entre 1 e 120 anos')
        }
        
        for field, (min_val, max_val, msg) in validations.items():
            try:
                value = float(data[field])
                if not (min_val <= value <= max_val):
                    return jsonify({
                        'error': f'Valor inválido para {field}',
                        'message': msg
                    }), 400
            except ValueError:
                return jsonify({
                    'error': f'Valor inválido para {field}',
                    'message': f'{field} deve ser um número'
                }), 400
        
        # ============================================
        # PASSO 4: PREPARAR DADOS
        # ============================================
        
        # Criar array com features na ordem correta
        features = [float(data[field]) for field in feature_names]
        features_array = np.array(features).reshape(1, -1)
        
        print(f"📊 Features preparadas: {features_array}")
        
        # ============================================
        # PASSO 5: NORMALIZAR DADOS
        # ============================================
        
        # Aplicar o mesmo escalonamento usado no treinamento
        features_scaled = scaler.transform(features_array)
        
        # ============================================
        # PASSO 6: FAZER PREDIÇÃO
        # ============================================
        
        # predict() retorna 0 (Sem diabetes) ou 1 (Com diabetes)
        prediction = model.predict(features_scaled)[0]
        
        # predict_proba() retorna [prob_sem_diabetes, prob_com_diabetes]
        probabilities = model.predict_proba(features_scaled)[0]
        probability_diabetes = float(probabilities[1])
        
        print(f"🎯 Predição: {prediction}")
        print(f"📊 Probabilidades: {probabilities}")
        
        # ============================================
        # PASSO 7: INTERPRETAR RESULTADOS
        # ============================================
        
        # Texto da predição
        prediction_text = "Com Diabetes" if prediction == 1 else "Sem Diabetes"
        
        # Nível de confiança
        if probability_diabetes >= 0.8 or probability_diabetes <= 0.2:
            confidence = "High"
        elif probability_diabetes >= 0.6 or probability_diabetes <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Nível de risco
        if probability_diabetes >= 0.7:
            risk_level = "Alto"
            risk_color = "danger"
        elif probability_diabetes >= 0.4:
            risk_level = "Médio"
            risk_color = "warning"
        else:
            risk_level = "Baixo"
            risk_color = "success"
        
        # ============================================
        # PASSO 8: GERAR RECOMENDAÇÕES
        # ============================================
        
        recommendations = []
        
        # Recomendações baseadas nos dados
        if data['Glucose'] > 140:
            recommendations.append({
                'category': 'Glicose',
                'message': 'Nível de glicose elevado. Consulte um médico.',
                'icon': '⚠️'
            })
        
        if data['BMI'] > 30:
            recommendations.append({
                'category': 'IMC',
                'message': 'IMC indica obesidade. Considere programa de perda de peso.',
                'icon': '🏃'
            })
        
        if data['BloodPressure'] > 90:
            recommendations.append({
                'category': 'Pressão',
                'message': 'Pressão arterial elevada. Monitore regularmente.',
                'icon': '💓'
            })
        
        if data['Age'] > 45 and probability_diabetes > 0.5:
            recommendations.append({
                'category': 'Idade',
                'message': 'Idade é fator de risco. Faça check-ups regulares.',
                'icon': '👨‍⚕️'
            })
        
        # Recomendações gerais
        if prediction == 1 or probability_diabetes > 0.5:
            recommendations.extend([
                {
                    'category': 'Consulta',
                    'message': 'Agende consulta com endocrinologista.',
                    'icon': '🏥'
                },
                {
                    'category': 'Exames',
                    'message': 'Solicite exame de hemoglobina glicada (HbA1c).',
                    'icon': '🧪'
                },
                {
                    'category': 'Estilo de vida',
                    'message': 'Adote dieta balanceada e exercícios regulares.',
                    'icon': '🥗'
                }
            ])
        else:
            recommendations.append({
                'category': 'Prevenção',
                'message': 'Mantenha hábitos saudáveis para prevenir diabetes.',
                'icon': '✅'
            })
        
        # ============================================
        # PASSO 9: PREPARAR RESPOSTA
        # ============================================
        
        response = {
            'prediction': prediction_text,
            'has_diabetes': bool(prediction),
            'probability': round(probability_diabetes, 4),
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probabilities': {
                'without_diabetes': round(float(probabilities[0]), 4),
                'with_diabetes': round(float(probabilities[1]), 4)
            },
            'recommendations': recommendations,
            'input_data': data,
            'interpretation': {
                'glucose_status': 'Normal' if data['Glucose'] < 140 else 'Elevado',
                'bmi_status': 'Normal' if data['BMI'] < 25 else ('Sobrepeso' if data['BMI'] < 30 else 'Obesidade'),
                'age_risk': 'Alto' if data['Age'] > 45 else 'Médio' if data['Age'] > 35 else 'Baixo'
            }
        }
        
        print(f"✅ Resposta preparada: {response['prediction']}")
        
        return jsonify(response), 200
    
    except ValueError as ve:
        return jsonify({
            'error': 'Erro de validação',
            'message': str(ve)
        }), 400
    
    except Exception as e:
        print(f"❌ Erro inesperado: {str(e)}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500

# ============================================
# INICIAR SERVIDOR
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 API FLASK PARA PREDIÇÃO DE DIABETES")
    print("="*70)
    
    if model is None:
        print("\n⚠️  ATENÇÃO: Modelo não carregado!")
        print("💡 Execute primeiro: python train_model.py")
    else:
        print("\n✅ Modelo carregado e pronto!")
    
    print("\n📡 Endpoints disponíveis:")
    print("   • http://localhost:5000/")
    print("   • http://localhost:5000/health")
    print("   • http://localhost:5000/model-info")
    print("   • http://localhost:5000/predict")
    print("="*70 + "\n")
    
    # Iniciar servidor Flask
    # debug=True: mostra erros detalhados e recarrega automático
    # host='0.0.0.0': permite acesso de outras máquinas
    # port=5000: porta onde a API escuta
    app.run(debug=True, host='0.0.0.0', port=5000)