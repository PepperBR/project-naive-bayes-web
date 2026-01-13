# Classificador de Acidentes de Trânsito

Aplicação web desenvolvida em Django para classificação de imagens de acidentes de trânsito utilizando algoritmo Naive Bayes.

## 📋 Visão Geral

O projeto consiste em duas partes principais:

1. **Treinamento do Modelo**: Notebook Jupyter que treina um classificador Naive Bayes para identificar três categorias de imagens:
   - Acidentes de trânsito graves
   - Acidentes de trânsito moderados
   - Não acidentes

2. **Aplicação Web**: Interface Django que permite upload de imagens e retorna a predição do modelo treinado.

## 🗂️ Estrutura do Projeto

```
project-naive-bayes-web/
├── naive_bayes_training/          # Treinamento do modelo
│   ├── train_model.ipynb          # Notebook de treinamento
│   └── dataset_finalized/        # Dataset com 720 imagens (240 por classe)
│
└── website/                       # Aplicação Django
    ├── manage.py
    ├── requirements.txt           # Dependências do projeto
    ├── config/                    # Configurações do Django
    ├── classifier/                # App de classificação
    │   ├── views.py              # Lógica de upload e predição
    │   ├── utils.py              # Carregamento do modelo e features
    │   ├── templates/            # HTMLs (index e result)
    │   └── static/               # CSS e imagens
    └── ml_models/                # Modelo e scaler salvos
        ├── modelo_ia.pkl
        └── scaler.pkl
```

## 🚀 Como Executar o Projeto

### Pré-requisitos

- Python 3.8+
- pip

### 1. Configurar o Ambiente Virtual

Navegue até a pasta `website`:

```powershell
cd website
```

Crie o ambiente virtual:

```powershell
python -m venv .venv
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar Dependências

```powershell
pip install -r requirements.txt
```

As dependências incluem:
- Django 6.0.1
- numpy 2.2.6
- opencv-python 4.12.0.88
- scikit-image 0.26.0
- scikit-learn 1.8.0
- Pillow 12.1.0

### 3. Executar o Servidor

```powershell
python manage.py runserver
```

Acesse a aplicação em: **http://127.0.0.1:8000/**

## 🧠 Fluxo de Funcionamento

### Fase 1: Treinamento do Modelo

1. **Dataset**: 720 imagens divididas em 3 classes (240 cada)
   - `dataset_final_severe_accident/`
   - `dataset_final_moderate_accident/`
   - `dataset_final_no_accident/`

2. **Extração de Features**: O notebook `train_model.ipynb` processa cada imagem e extrai 8112 características:
   - **HOG** (8100 features): Histogram of Oriented Gradients para detecção de formas
   - **Canny** (1 feature): Densidade de bordas
   - **Harris** (1 feature): Densidade de cantos
   - **LBP** (10 features): Local Binary Pattern para textura

3. **Pré-processamento**:
   - Redimensionamento para 128x128 pixels
   - Conversão para escala de cinza
   - Normalização com StandardScaler

4. **Treinamento**:
   - Algoritmo: Gaussian Naive Bayes
   - Split: 80% treino / 20% teste
   - Acurácia alcançada: ~76%

5. **Salvamento**:
   - `modelo_ia.pkl`: Modelo treinado
   - `scaler.pkl`: StandardScaler ajustado

### Fase 2: Aplicação Web

1. **Upload de Imagem**:
   - Usuário acessa a página inicial
   - Seleciona uma imagem para análise
   - Clica em "Prever"

2. **Processamento** (`classifier/utils.py`):
   - Carrega modelo e scaler (lazy loading)
   - Lê imagem em escala de cinza
   - Extrai 8112 features (mesmo pipeline do treinamento)
   - Normaliza features com o scaler
   - Faz predição com o modelo

3. **Resultado**:
   - Classe predita é mapeada para texto legível
   - Exibe resultado na página de resultados

## 🎯 Detalhes Técnicos

### Extração de Features

```python
# Mesma função usada no treinamento e na predição
def extrair_features_avancadas(img_array):
    img = cv2.resize(img_array, (128, 128))
    
    # HOG
    features_hog = hog(img, orientations=9, pixels_per_cell=(8,8),
                       cells_per_block=(2,2))
    
    # Bordas (Canny)
    edges = cv2.Canny(img, 100, 200)
    densidade_bordas = [np.sum(edges > 0) / edges.size]
    
    # Cantos (Harris)
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    densidade_cantos = [np.sum(dst > 0.01 * dst.max()) / dst.size]
    
    # Textura (LBP)
    lbp = local_binary_pattern(img, 8, 1, method="uniform")
    hist_lbp = np.histogram(lbp.ravel(), bins=10, range=(0,10), density=True)[0]
    
    return np.hstack([features_hog, densidade_bordas, densidade_cantos, hist_lbp])
```

### Mapeamento de Classes

| Valor | Classe |
|-------|--------|
| 0 | Acidente de trânsito grave |
| 1 | Acidente de trânsito moderado |
| 2 | Não é acidente |

## 📊 Métricas de Desempenho

- **Precisão Média**: 76%
- **Total de Features**: 8112
- **Tempo de Predição**: ~1-2 segundos por imagem

## 🔄 Retreinamento do Modelo

Para retreinar o modelo com novos dados:

1. Adicione imagens nas pastas do dataset em `naive_bayes_training/dataset_finalized/`
2. Abra o notebook `train_model.ipynb`
3. Execute todas as células sequencialmente
4. Os arquivos `modelo_ia.pkl` e `scaler.pkl` serão atualizados em `website/ml_models/`
5. Reinicie o servidor Django para carregar o novo modelo

## 🛠️ Tecnologias Utilizadas

- **Backend**: Django 6.0.1
- **Machine Learning**: scikit-learn (Gaussian Naive Bayes)
- **Processamento de Imagens**: OpenCV, scikit-image
- **Frontend**: HTML5, CSS3
- **Ambiente de Treinamento**: Jupyter Notebook

## Equipe

- Hiel Saraiva
- Roberta Alanis
- João Marcelo Pimenta
- Ryan Leite
- Ruan Venâncio
