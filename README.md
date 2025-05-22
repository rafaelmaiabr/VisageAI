# VisageAI - Análise de Emoções em Tempo Real

VisageAI é uma aplicação de análise de emoções em tempo real que utiliza visão computacional e inteligência artificial para detectar e analisar expressões faciais, tanto através da webcam quanto da captura de tela.

## 🚀 Características

- **Análise em Tempo Real**: Detecta e analisa emoções instantaneamente
- **Múltiplos Modos de Captura**:
  - Modo Webcam para análise pessoal
  - Modo Captura de Tela para análise de videoconferências
- **Interface em Português**: Todas as emoções e controles são exibidos em português
- **Emoções Detectadas**:
  - Raiva
  - Nojo
  - Medo
  - Felicidade
  - Tristeza
  - Surpresa
  - Neutro

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Webcam (para o modo câmera)
- Sistema operacional Windows

## 🔧 Instalação

### Método 1: Usando o start.bat (Recomendado)

1. Clone ou baixe este repositório
2. Execute o arquivo `start.bat`
   - O script irá automaticamente configurar o ambiente virtual
   - Instalar todas as dependências necessárias
   - Iniciar a aplicação

### Método 2: Instalação Manual

1. Clone ou baixe este repositório
2. Abra o terminal na pasta do projeto
3. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```
4. Ative o ambiente virtual:
   ```bash
   .\venv\Scripts\activate
   ```
5. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
6. Execute a aplicação:
   ```bash
   python main.py
   ```

## 🎮 Como Usar

1. Inicie o programa usando um dos métodos acima
2. Use os seguintes controles:
   - Pressione 'c' para alternar entre modo webcam e captura de tela
   - Pressione 'q' para sair do programa

### Modo Webcam
- Análise direta através da sua webcam
- Ideal para análise pessoal de emoções

### Modo Captura de Tela
- Análise de qualquer parte da sua tela
- Perfeito para videoconferências (Google Meet, Zoom, etc.)
- Detecta múltiplos rostos simultaneamente

## 📊 Visualização
- Exibe a emoção predominante
- Mostra percentuais para cada emoção detectada
- Indica o modo atual de captura (Webcam/Tela)
- Desenha retângulos ao redor dos rostos detectados

pip install tf-keras