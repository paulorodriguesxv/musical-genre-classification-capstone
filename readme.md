# Nanodegree Engenheiro de Machine Learning
## Projeto final
### Classificação de áudio por gêneros musicais
Paulo Leonardo Vieira Rodrigues  
10 de maio de 2018

-------

## Requisitos:
Este projeto foi desenvolvido utilizandos os seguintes programas de computador e/ou bibliotecas python

    - python 3.6
    - jupter notebook
    - librosa == 0.6
    - scikit-learn == 0.19.1
    - scipy==1.0.0
    - seaborn==0.8.1
    - numpy==1.14.2
    - pandas==0.22.0
    - matplotlib==2.1.2

Junto ao repositório do projeto há um arquivo chamado requirements.txt, nele há todas as depêndencias necessárias para
rodar o projeto. Para instalá-las, basta executar

```python
  pip install -r requirements.txt
```
-------
## Dataset:
O dataset utilizado nesse projeto é o dataset GTZAN, que é um dataset contendo uma coleção musical dividida em dez gêneros musicais Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae e Rock). Cada gênero apresenta 100 amostras de áudios com 30 segundos de duração.
Este dataset pode ser obtido em: http://opihi.cs.uvic.ca/sound/genres.tar.gz

-------
## Codigo:
Todo o projeto foi desenvolvido passo-a-passo utilizando o arquivo **Classificação de áudio por gêneros musicais.ipynb**
Nele estão presentes todos os passos para execução do projeto. Este arquivo é um jupyther notebook
Todos os links para bibliotecas e datasets estão presentes na sessão "referências" do notebook do projeto. 
Para executar o notebook:

```python
 python notebook Classificação\ de\ áudio\ por\ gêneros\ musicais.ipynb 
``` 

O projeto também utiliza alguns arquivos python para auxiliar no processamento de gráficos, leitura de arquivos
e geração de dataframes. Estes arquivos encontram-se na pasta **helpers**

-------
## Relatório:

O relatório final em pdf está nomeado como report.pdf e é uma compilação do que foi desenvolvido no notebook