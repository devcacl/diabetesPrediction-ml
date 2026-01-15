/**
 * 🏥 Aplicação React para Predição de Diabetes
 * 
 * COMO FUNCIONA:
 * 1. Usuário preenche dados médicos no formulário
 * 2. Clica em "Analisar Risco"
 * 3. React envia dados para API Flask
 * 4. API retorna predição
 * 5. React exibe resultado de forma visual
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // ============================================
  // ESTADO DO COMPONENTE
  // ============================================
  
  /**
   * useState é um Hook que gerencia estado em React
   * Formato: const [valor, função_para_mudar] = useState(valor_inicial)
   */
  
  // Dados do formulário (valores iniciais realistas)
  const [formData, setFormData] = useState({
    'Pregnancies': 3,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 20,
    'Insulin': 80,
    'BMI': 26.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 45
  });
  
  // Resultado da predição
  const [prediction, setPrediction] = useState(null);
  
  // Estado de carregamento
  const [loading, setLoading] = useState(false);
  
  // Mensagem de erro
  const [error, setError] = useState(null);
  
  // Status da API
  const [apiOnline, setApiOnline] = useState(false);
  
  // ============================================
  // EFECTO: VERIFICAR API AO INICIAR
  // ============================================
  
  /**
   * useEffect executa código quando componente carrega
   * [] significa "executar apenas uma vez"
   */
  useEffect(() => {
    checkAPIHealth();
  }, []);
  
  /**
   * Verifica se a API está online
   */
  const checkAPIHealth = async () => {
    try {
      const response = await axios.get('http://localhost:5000/health');
      setApiOnline(response.data.status === 'healthy');
      console.log('✅ API está online!');
    } catch (err) {
      setApiOnline(false);
      console.error('❌ API offline:', err);
    }
  };
  
  // ============================================
  // FUNÇÕES DO FORMULÁRIO
  // ============================================
  
  /**
   * Atualiza estado quando usuário digita no formulário
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Atualizar apenas o campo que mudou
    setFormData(prevData => ({
      ...prevData,      // Mantém outros campos
      [name]: parseFloat(value)  // Atualiza campo específico
    }));
  };
  
  /**
   * Função chamada ao enviar formulário
   */
  const handleSubmit = async (e) => {
    e.preventDefault();  // Previne recarregar página
    
    // Resetar estados
    setLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      console.log('📤 Enviando dados para API:', formData);
      
      // Fazer requisição POST para API
      const response = await axios.post(
        'http://localhost:5000/predict',
        formData,
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      console.log('📥 Resposta recebida:', response.data);
      
      // Salvar resultado
      setPrediction(response.data);
      
    } catch (err) {
      // Tratar erros
      if (err.response) {
        // API retornou erro
        setError(err.response.data.message || 'Erro ao fazer predição');
      } else if (err.request) {
        // Sem resposta da API
        setError('Não foi possível conectar com a API. Ela está rodando?');
      } else {
        // Outro erro
        setError('Erro inesperado: ' + err.message);
      }
      console.error('❌ Erro:', err);
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Limpa o formulário
   */
  const handleReset = () => {
    setFormData({
      'Pregnancies': 3,
      'Glucose': 120,
      'BloodPressure': 70,
      'SkinThickness': 20,
      'Insulin': 80,
      'BMI': 26.5,
      'DiabetesPedigreeFunction': 0.5,
      'Age': 45
    });
    setPrediction(null);
    setError(null);
  };
  
  // ============================================
  // RENDERIZAÇÃO DA INTERFACE
  // ============================================
  
  return (
    <div className="App">
      
      {/* CABEÇALHO */}
      <header className="app-header">
        <div className="header-content">
          <h1>🏥 Sistema de Predição de Diabetes</h1>
          <p>Análise de risco baseada em Machine Learning (XGBoost)</p>
        </div>
        
        {/* Indicador de status da API */}
        <div className={`api-status ${apiOnline ? 'online' : 'offline'}`}>
          <span className="status-dot"></span>
          API: {apiOnline ? 'Online' : 'Offline'}
        </div>
      </header>
      
      {/* CONTAINER PRINCIPAL */}
      <div className="container">
        
        {/* COLUNA ESQUERDA: FORMULÁRIO */}
        <div className="form-section">
          <h2>📋 Dados do Paciente</h2>
          <p className="subtitle">Preencha as informações médicas abaixo</p>
          
          <form onSubmit={handleSubmit}>
            
            {/* Gestações */}
            <div className="form-group">
              <label>
                <span className="label-icon">🤰</span>
                Gestações:
                <span className="label-info">Número de gestações anteriores</span>
              </label>
              <input
                type="number"
                name="Pregnancies"
                value={formData.Pregnancies}
                onChange={handleInputChange}
                min="0"
                max="20"
                step="1"
                required
              />
            </div>
            
            {/* Glicose */}
            <div className="form-group">
              <label>
                <span className="label-icon">🩸</span>
                Glicose (mg/dL):
                <span className="label-info">Nível de glicose no sangue</span>
              </label>
              <input
                type="number"
                name="Glucose"
                value={formData.Glucose}
                onChange={handleInputChange}
                min="0"
                max="300"
                step="1"
                required
              />
              <small className="input-hint">
                Normal: 70-100 | Pré-diabetes: 100-125 | Diabetes: 126+
              </small>
            </div>
            
            {/* Pressão Arterial */}
            <div className="form-group">
              <label>
                <span className="label-icon">💓</span>
                Pressão Arterial (mm Hg):
                <span className="label-info">Pressão arterial diastólica</span>
              </label>
              <input
                type="number"
                name="BloodPressure"
                value={formData.BloodPressure}
                onChange={handleInputChange}
                min="0"
                max="200"
                step="1"
                required
              />
              <small className="input-hint">Normal: 60-80</small>
            </div>
            
            {/* Espessura da Pele */}
            <div className="form-group">
              <label>
                <span className="label-icon">📏</span>
                Espessura da Pele (mm):
                <span className="label-info">Medida do tríceps</span>
              </label>
              <input
                type="number"
                name="SkinThickness"
                value={formData.SkinThickness}
                onChange={handleInputChange}
                min="0"
                max="100"
                step="1"
                required
              />
            </div>
            
            {/* Insulina */}
            <div className="form-group">
              <label>
                <span className="label-icon">💉</span>
                Insulina (mu U/ml):
                <span className="label-info">Nível de insulina no sangue</span>
              </label>
              <input
                type="number"
                name="Insulin"
                value={formData.Insulin}
                onChange={handleInputChange}
                min="0"
                max="900"
                step="1"
                required
              />
            </div>
            
            {/* IMC */}
            <div className="form-group">
              <label>
                <span className="label-icon">⚖️</span>
                IMC (Índice de Massa Corporal):
                <span className="label-info">Peso (kg) / Altura² (m)</span>
              </label>
              <input
                type="number"
                name="BMI"
                value={formData.BMI}
                onChange={handleInputChange}
                min="0"
                max="70"
                step="0.1"
                required
              />
              <small className="input-hint">
                Normal: 18.5-24.9 | Sobrepeso: 25-29.9 | Obesidade: 30+
              </small>
            </div>
            
            {/* Histórico Familiar */}
            <div className="form-group">
              <label>
                <span className="label-icon">👨‍👩‍👧‍👦</span>
                Histórico Familiar:
                <span className="label-info">Função de pedigree do diabetes</span>
              </label>
              <input
                type="number"
                name="DiabetesPedigreeFunction"
                value={formData.DiabetesPedigreeFunction}
                onChange={handleInputChange}
                min="0"
                max="3"
                step="0.001"
                required
              />
              <small className="input-hint">Valor entre 0 e 3</small>
            </div>
            
            {/* Idade */}
            <div className="form-group">
              <label>
                <span className="label-icon">🎂</span>
                Idade (anos):
              </label>
              <input
                type="number"
                name="Age"
                value={formData.Age}
                onChange={handleInputChange}
                min="1"
                max="120"
                step="1"
                required
              />
            </div>
            
            {/* Botões */}
            <div className="button-group">
              <button 
                type="submit" 
                className="btn-primary"
                disabled={loading || !apiOnline}
              >
                {loading ? '🔄 Analisando...' : '🔬 Analisar Risco'}
              </button>
              
              <button 
                type="button" 
                onClick={handleReset}
                className="btn-secondary"
              >
                🔄 Limpar
              </button>
            </div>
            
            {!apiOnline && (
              <div className="warning-message">
                ⚠️ API offline. Execute: python backend/app.py
              </div>
            )}
            
          </form>
        </div>
        
        {/* COLUNA DIREITA: RESULTADOS */}
        <div className="results-section">
          
          {/* ERRO */}
          {error && (
            <div className="error-message">
              <h3>❌ Erro</h3>
              <p>{error}</p>
            </div>
          )}
          
          {/* RESULTADO */}
          {prediction && (
            <div className={`prediction-result ${prediction.risk_color}`}>
              <h2>📊 Resultado da Análise</h2>
              
              {/* Badge principal */}
              <div className="prediction-badge">
                <span className="prediction-icon">
                  {prediction.has_diabetes ? '⚠️' : '✅'}
                </span>
                <span className="prediction-text">
                  {prediction.prediction}
                </span>
              </div>
              
              {/* Risco */}
              <div className="risk-section">
                <h3>Nível de Risco</h3>
                <div className={`risk-badge ${prediction.risk_color}`}>
                  {prediction.risk_level}
                </div>
              </div>
              
              {/* Probabilidade */}
              <div className="probability-section">
                <h3>Probabilidade de Diabetes</h3>
                <div className="probability-bar-container">
                  <div 
                    className="probability-bar"
                    style={{ width: `${prediction.probability * 100}%` }}
                  ></div>
                </div>
                <p className="probability-text">
                  {(prediction.probability * 100).toFixed(1)}%
                </p>
              </div>
              
              {/* Confiança */}
              <div className="confidence-section">
                <span className={`confidence-badge ${prediction.confidence.toLowerCase()}`}>
                  Confiança: {prediction.confidence === 'High' ? 'Alta' : 
                             prediction.confidence === 'Medium' ? 'Média' : 'Baixa'}
                </span>
              </div>
              
              {/* Interpretação */}
              <div className="interpretation-section">
                <h3>📋 Interpretação dos Dados</h3>
                <div className="interpretation-grid">
                  <div className="interpretation-item">
                    <span className="interpretation-label">Glicose:</span>
                    <span className={`interpretation-value ${
                      prediction.interpretation.glucose_status === 'Normal' ? 'good' : 'bad'
                    }`}>
                      {prediction.interpretation.glucose_status}
                    </span>
                  </div>
                  <div className="interpretation-item">
                    <span className="interpretation-label">IMC:</span>
                    <span className={`interpretation-value ${
                      prediction.interpretation.bmi_status === 'Normal' ? 'good' : 'bad'
                    }`}>
                      {prediction.interpretation.bmi_status}
                    </span>
                  </div>
                  <div className="interpretation-item">
                    <span className="interpretation-label">Risco por Idade:</span>
                    <span className={`interpretation-value ${
                      prediction.interpretation.age_risk === 'Baixo' ? 'good' : 'warning'
                    }`}>
                      {prediction.interpretation.age_risk}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Recomendações */}
              <div className="recommendations-section">
                <h3>💡 Recomendações</h3>
                <div className="recommendations-list">
                  {prediction.recommendations.map((rec, index) => (
                    <div key={index} className="recommendation-item">
                      <span className="rec-icon">{rec.icon}</span>
                      <div className="rec-content">
                        <strong>{rec.category}:</strong>
                        <p>{rec.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Detalhes técnicos */}
              <details className="technical-details">
                <summary>🔬 Detalhes Técnicos</summary>
                <table>
                  <tbody>
                    <tr>
                      <td>Probabilidade Sem Diabetes:</td>
                      <td>{(prediction.probabilities.without_diabetes * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Probabilidade Com Diabetes:</td>
                      <td>{(prediction.probabilities.with_diabetes * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Algoritmo:</td>
                      <td>XGBoost</td>
                    </tr>
                    <tr>
                      <td>Confiança:</td>
                      <td>{prediction.confidence}</td>
                    </tr>
                  </tbody>
                </table>
              </details>
            </div>
          )}
          
          {/* MENSAGEM INICIAL */}
          {!prediction && !error && !loading && (
            <div className="info-message">
              <h3>ℹ️ Como Funciona</h3>
              <p>
                Este sistema usa <strong>Machine Learning (XGBoost)</strong> para 
                analisar dados médicos e prever o risco de diabetes.
              </p>
              
              <div className="algorithm-info">
                <h4>🧠 Sobre o XGBoost</h4>
                <p>
                  <strong>XGBoost</strong> (Extreme Gradient Boosting) é um dos 
                  algoritmos mais poderosos para classificação. Ele funciona criando 
                  múltiplas árvores de decisão que trabalham juntas para fazer 
                  predições precisas.
                </p>
                <ul>
                  <li>✅ Alta precisão em dados médicos</li>
                  <li>✅ Lida bem com dados desbalanceados</li>
                  <li>✅ Explica importância de cada variável</li>
                  <li>✅ Usado por hospitais e pesquisadores</li>
                </ul>
              </div>
              
              <div className="steps-info">
                <h4>📝 Passos para Usar</h4>
                <ol>
                  <li>Preencha todos os campos com dados médicos reais</li>
                  <li>Clique em "Analisar Risco"</li>
                  <li>Veja a predição e recomendações</li>
                  <li>Consulte um médico para avaliação completa</li>
                </ol>
              </div>
              
              <div className="disclaimer">
                <strong>⚠️ Aviso Importante:</strong>
                <p>
                  Este é um sistema educacional e NÃO substitui consulta médica. 
                  Sempre consulte profissionais de saúde qualificados.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* RODAPÉ */}
      <footer className="app-footer">
        <p>
          🏥 Sistema de Predição de Diabetes | 
          Algoritmo: XGBoost | 
          Backend: Flask + Python | 
          Frontend: React
        </p>
        <p>
          ⚠️ Este sistema é apenas educacional e não substitui orientação médica profissional
        </p>
      </footer>
      
    </div>
  );
}

export default App;