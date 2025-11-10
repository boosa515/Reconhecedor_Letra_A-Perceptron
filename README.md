# < Reconhecedor da Letra A (Perceptron) > 🧠
<br/>

<br/>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Badge"/>
  <img src="https://img.shields.io/badge/PyQt-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt Badge"/>
</p>
<br/>

<br/>

<br/>

## Download
<p align=>
  <strong>Código-Fonte:</strong> <a href="[https://github.com/SEU-USUARIO/SEU-REPOSITORIO/archive/refs/heads/main.zip](https://github.com/boosa515/Reconhecedor-Letra-A---Perceptron/archive/refs/heads/main.zip)"><strong>Clique Aqui (ZIP)</strong></a>
</p>
<br/>

## 💡 Sobre o Projeto

Este é um reconhecedor de caracteres para a letra 'A' desenvolvido em **Python**. O "cérebro" do projeto é um **Perceptron** (a forma mais simples de rede neural) construído com **PyTorch**, e a interface gráfica (GUI) foi feita com **PyQt5**.

Este projeto foi desenvolvido como atividade para a disciplina de **Inteligência Artificial**, do curso de Engenharia da Computação.

O principal diferencial do projeto é o sistema de **Aprendizado Contínuo** (ou *Online Learning*). O modelo é treinado em um grande conjunto de dados (EMNIST) para criar uma base (v5.0), e o usuário pode então testá-lo com suas próprias imagens ou desenhos. Se o Perceptron errar, o usuário pode clicar em **"Ele Errou!"**, o que **retreina e salva o modelo instantaneamente** com essa nova informação, melhorando sua precisão a cada uso.
<br/>

<br/>

## ⚙️ Principais Funcionalidades

* **Modelo Perceptron (PyTorch):** Um classificador binário treinado para diferenciar "A" de "Não-A". O modelo foi "tunado" (v5.0) para encontrar o melhor equilíbrio entre precisão (*precision*) e sensibilidade (*recall*).
* **Duas Formas de Teste:**
    * **Carregar Imagem:** O usuário pode testar qualquer arquivo de imagem do seu computador.
    * **Desenhar na Tela:** Um canvas de desenho permite ao usuário desenhar a letra 'A' (ou outra) com o mouse.
* **Aprendizado por Feedback:** A interface possui botões de "Ele Acertou!" e "Ele Errou!", permitindo ao usuário corrigir o modelo em tempo real. O cérebro (`perceptron_A_v5.pth`) é atualizado a cada correção.
* **Interface Moderna:** A GUI possui um sistema de "página única" para alternar entre as telas de início e desenho.
* **Tema Claro/Escuro:** Um botão no canto superior alterna o tema da aplicação.
<br/>
<br/>

## Pré-requisitos

* Python 3.x

<br/>

<br/>

# 1. Configurar o Ambiente

Assumindo que você já clonou o repositório e está no diretório do projeto:

  Cria e ativa o ambiente virtual
  
```bash
python -m venv .venv
```
  
Windows:
```bash
.\venv\Scripts\Activate
```
  
macOS/Linux:
```bash
source .venv/bin/activate
```
  
Instala as dependências
```bash
  pip install torch torchvision scikit-learn PyQt5 Pillow
```
<br/>

# 2. Rodar a Aplicação
O projeto funciona em duas etapas: primeiro treinamos o modelo base, depois executamos a interface.

  1. Treinar o modelo v5.0 (Isto criará o arquivo `perceptron_A_v5.pth`)
```bash
python treinar_modelo.py
```
  
2. Iniciar a aplicação
```bash
python testar_gui.py
```
  
## Acesso

A janela principal do reconhecedor será aberta automaticamente.
