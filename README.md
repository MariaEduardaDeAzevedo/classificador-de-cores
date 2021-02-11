# Detector de cores em imagens!

Já pensou em mostrar um objeto para a câmera e saber a cor que ele é? Esse projeto trabalha com essa ideia!

O código é escrito em [Python3.8.5](https://www.python.org/) e utiliza das bibliotecas [scikit-learn](https://scikit-learn.org/) e [OpenCV](https://opencv.org/) para processar imagens, extrair características e treinar um modelo de classificação, utilizando o algoritmo [K-Nearest Neighbors](https://medium.com/brasil-ai/knn-k-nearest-neighbors-1-e140c82e9c4e), que consegue identificar a cor em predominância na imagem.

De modo geral, as características das imagens são extraídas partindo do [histograma](https://ferramentasdaqualidade.org/histograma/) de cada um dos canais de cor (R, ou vermelho, G, ou verde e B, ou azul). Com isso um array com todas as saídas de cada um dos histogramas é montado e utiliza-se da [Análise de Componentes Principais (PCA)](https://operdata.com.br/blog/analise-de-componentes-principais/) para a extração das características passadas para o treinamento do modelo e também na predição.

Com isso, o modelo consegue reconhecer os padrões dos canais das imagens para classificá-las em 5 diferentes cores:

- Vermelho
- Amarelo
- Verde
- Azul
- Laranja 

### Como faço para rodar?

Para rodar esse código na sua máquina é simples! Segue o tutorial:

1. **Clone esse repositório (ou dê fork e clone!)**
    ```
    git clone https://github.com/MariaEduardaDeAzevedo/classificador-de-cores.git
    ```
2. **Vá até o diretório clonado...**
    ```
    cd classificador-de-cores.git
    ```
3. **Crie um ambiente virtual!**
    ```
    python3 -m venv env
    ```
4. **Inicie o ambiente vitual...**
    ```
    source env/bin/activate
    ```
5. **Instale o requirements.txt...**
    ```
    pip install -r requirements.txt
    ```
    ou
    ```
    pip3 install -r requirements.txt
    ```
6. **Rode o script open_cam.py**
    ```
    python3 open_cam.py
    ```
7. **Se divirta!**