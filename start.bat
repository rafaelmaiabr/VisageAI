@echo off
echo Inicializando VisageAI...

REM Verifica se o Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo Python nao encontrado! Por favor, instale o Python 3.8 ou superior.
    pause
    exit
)

REM Verifica se o ambiente virtual existe
if not exist venv (
    echo Criando ambiente virtual...
    python -m venv venv
)

REM Ativa o ambiente virtual
call venv\Scripts\activate.bat

REM Instala/Atualiza as dependências
echo Instalando/Atualizando dependencias...
pip install -r requirements.txt

REM Executa o programa
echo Iniciando a aplicacao...
python main.py

REM Desativa o ambiente virtual ao finalizar
deactivate

pause
