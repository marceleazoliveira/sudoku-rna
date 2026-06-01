# Resolução de Sudoku com Rede Neural Artificial Multicamadas

## 1. Introdução

Este trabalho propõe uma solução baseada em Rede Neural Artificial (RNA) multicamadas para auxiliar na resolução e validação do quebra-cabeça Sudoku. Inicialmente, o problema foi tratado em uma grade 4x4, com subgrupos 2x2 e símbolos pertencentes ao conjunto S = {1, 2, 3, 4}. Em seguida, a abordagem foi generalizada para grades NxN, como o caso 9x9.

O objetivo principal foi demonstrar como uma RNA pode aprender a reconhecer padrões de validade em tabuleiros de Sudoku, enquanto a geração da solução final é garantida por regras explícitas do problema e por um algoritmo de busca com restrições.

## 2. Regras do Sudoku

O Sudoku é um problema de satisfação de restrições. Para que um tabuleiro seja considerado válido, ele deve atender às seguintes condições:

1. Cada célula deve conter apenas um número pertencente ao conjunto permitido.
2. Nenhuma linha pode conter números repetidos.
3. Nenhuma coluna pode conter números repetidos.
4. Nenhum subgrupo pode conter números repetidos.
5. A solução final deve preencher todas as células respeitando as restrições anteriores.

No caso do Sudoku 4x4, a grade possui 16 células e subgrupos 2x2. Já no Sudoku 9x9, a grade possui 81 células e subgrupos 3x3.

## 3. Organização do Projeto

O projeto foi organizado com dois códigos principais:

```text
sudoku-rna/
├── codigo_4x4/
│   └── sudoku4_rna.py
├── codigo_nxn/
│   └── sudoku_n_rna.py
├── imagens/
│   ├── 4x4/
│   └── nxn/
└── relatorio.md
```

O arquivo `sudoku4_rna.py` contém a implementação inicial para o Sudoku 4x4. Esse código foi usado como base para demonstrar o funcionamento completo da RNA em um problema menor, onde é possível gerar todas as soluções válidas.

O arquivo `sudoku_n_rna.py` contém a versão generalizada para grades NxN. Essa versão permite testar grades maiores, como 9x9, mas apresenta limitações práticas devido ao crescimento do espaço de busca e da quantidade de combinações possíveis.

A pasta `imagens/` armazena os tabuleiros iniciais e resolvidos gerados durante a execução dos códigos. Essas imagens são utilizadas no relatório para demonstrar visualmente os resultados obtidos.

## 4. Implementação para Sudoku 4x4

A primeira implementação foi feita para uma grade 4x4 com subgrupos 2x2. Nesse caso, o conjunto de símbolos utilizado foi:

```text
S = {1, 2, 3, 4}
```

A RNA foi treinada para reconhecer se um tabuleiro completo é válido ou inválido. Para isso, foram geradas todas as soluções completas válidas do Sudoku 4x4.

A entrada da rede representa o tabuleiro codificado numericamente, e a saída indica a probabilidade de o tabuleiro ser válido.

Além da RNA, foi utilizado um algoritmo de busca com restrições para preencher tabuleiros incompletos. Assim, a RNA atua como uma ferramenta de reconhecimento, enquanto a validade final da solução é garantida pelas regras formais do Sudoku.

## 5. Conjunto de Dados

Para o Sudoku 4x4, foram geradas 288 soluções completas válidas. A partir dessas soluções, foram criadas amostras positivas e negativas para treinamento e teste.

As amostras positivas correspondem a tabuleiros válidos, enquanto as amostras negativas são geradas a partir de alterações nos tabuleiros, criando violações nas regras de linha, coluna ou subgrupo.

Exemplo de resultado obtido:

```text
Soluções completas válidas 4x4 geradas: 288
Amostras de treino: 1843
Amostras de teste : 461

Acurácia treino: 1.0000
Acurácia teste : 0.9978
```

Esses resultados indicam que, para o caso 4x4, a RNA conseguiu aprender muito bem os padrões de validade do Sudoku.

## 6. Exemplo de Execução 4x4

Um exemplo de tabuleiro inicial gerado aleatoriamente é apresentado abaixo:

```text
[[2 0 1 0]
 [1 3 2 0]
 [3 0 0 0]
 [0 0 3 2]]
```

Nesse tabuleiro, o valor 0 representa uma célula vazia.

A solução final encontrada foi:

```text
[[2 4 1 3]
 [1 3 2 4]
 [3 2 4 1]
 [4 1 3 2]]
```

A solução foi considerada válida pelas regras do Sudoku e também foi reconhecida pela RNA com alta probabilidade.

Exemplo de imagem gerada:

```markdown
![Tabuleiro inicial 4x4](imagens/4x4/sudoku_inicial_1.png)

![Tabuleiro resolvido 4x4](imagens/4x4/sudoku_resolvido_1.png)
```

## 7. Generalização para Sudoku NxN

Após a implementação do Sudoku 4x4, o código foi generalizado para trabalhar com grades NxN. A principal mudança foi substituir os valores fixos da grade por uma configuração variável.

Na versão 4x4, o código trabalhava com valores fixos como:

```python
N = 4
BOX = 2
S = [1, 2, 3, 4]
```

Na versão generalizada, esses valores passaram a depender do tamanho escolhido para a grade. Assim, o programa pode ser executado, por exemplo, com:

```bash
python sudoku_n_rna.py --n 4
```

ou:

```bash
python sudoku_n_rna.py --n 9 --solucoes 200 --epocas 80 --vazios 20
```

Para uma grade 9x9, os subgrupos são 3x3. Para uma grade 16x16, os subgrupos seriam 4x4.

## 8. Mudanças Necessárias na Generalização

A generalização do Sudoku 4x4 para NxN exigiu várias alterações importantes:

### 8.1 Tamanho da Grade

O código deixou de depender de uma grade fixa 4x4 e passou a utilizar uma variável `N`, representando o tamanho da grade.

### 8.2 Tamanho dos Subgrupos

No Sudoku 4x4, os subgrupos são 2x2. No Sudoku 9x9, são 3x3. Na generalização inicial, foi usada a relação:

```text
subgrupo = √N x √N
```

Essa abordagem funciona para grades como:

```text
4x4  -> 2x2
9x9  -> 3x3
16x16 -> 4x4
```



### 8.3 Codificação da Entrada da RNA

Com o aumento da grade, a entrada da RNA também cresce. No 4x4, existem apenas 16 células. No 9x9, existem 81 células. No 16x16, existem 256 células.

Isso torna a rede mais difícil de treinar e exige mais dados.

### 8.4 Geração do Conjunto de Dados

No Sudoku 4x4, foi possível gerar todas as 288 soluções válidas. Porém, para o Sudoku 9x9, o número de soluções possíveis é extremamente grande. Dessa forma, não é viável enumerar todas as soluções.

Na versão NxN, foi necessário usar amostragem, isto é, gerar apenas uma quantidade limitada de soluções válidas para formar o conjunto de treinamento.

### 8.5 Tempo de Execução

Quanto maior a grade, maior o tempo necessário para gerar soluções, criar amostras, treinar a RNA e resolver os tabuleiros incompletos.

Por isso, no caso 9x9, foram usados parâmetros menores, como:

```bash
python sudoku_n_rna.py --n 9 --solucoes 200 --epocas 80 --vazios 20
```

## 9. Resultados Obtidos para 9x9

Na execução com Sudoku 9x9, foram usadas 200 soluções completas como base para geração do conjunto de dados.

Resultado obtido:

```text
Tamanho da grade: 9x9
Subgrupos: 3x3
Soluções completas usadas como base: 200
Amostras de treino: 960
Amostras de teste : 240

Acurácia treino: 0.9010
Acurácia teste : 0.3125
```

Apesar de a RNA ter alcançado 90,10% de acurácia no treinamento, sua acurácia no teste foi de apenas 31,25%. Isso indica que a rede não conseguiu generalizar bem para novos exemplos.

Esse comportamento caracteriza um possível overfitting, pois a RNA aprendeu padrões específicos do conjunto de treino, mas não conseguiu reconhecer adequadamente tabuleiros diferentes no conjunto de teste.

Mesmo assim, as soluções finais foram consideradas válidas pelas regras do Sudoku, pois o resolvedor utiliza busca com restrições.

## 10. Discussão Sobre as Dificuldades da Generalização

A principal dificuldade da generalização para NxN está no crescimento combinatório do problema. No Sudoku 4x4, o espaço de busca já é grande, mas ainda controlável. No Sudoku 9x9, a quantidade de combinações possíveis cresce drasticamente.

Se o problema fosse resolvido apenas por geração aleatória de tabuleiros e teste de validade, a abordagem seria ineficiente. Isso ocorre porque a maioria das combinações possíveis não satisfaz as regras do Sudoku.

Por exemplo, para uma grade NxN, o número bruto de combinações possíveis é:

```text
N^(N*N)
```

No caso 4x4:

```text
4^16 = 4.294.967.296 combinações possíveis
```

Mesmo assim, apenas 288 correspondem a soluções válidas completas.

Isso mostra que o Sudoku não deve ser tratado apenas como um problema de tentativa e erro, mas sim como um problema de raciocínio baseado em restrições.

A RNA consegue aprender padrões a partir dos dados, mas não garante sozinha que todas as regras do Sudoku serão respeitadas. Por isso, a solução proposta combina RNA com regras explícitas e busca.

## 11. Limitações da Abordagem

A solução apresenta algumas limitações:

1. A RNA reconhece padrões de validade, mas não é responsável sozinha por resolver o Sudoku.
2. Para grades maiores, como 9x9 e 16x16, a geração de dados se torna mais difícil.
3. A rede pode sofrer overfitting quando o conjunto de treinamento é pequeno.
4. A versão NxN inicial suporta melhor grades com subgrupos quadrados, como 4x4, 9x9 e 16x16.
5. Para grades como 6x6 e 8x8, seria necessário adaptar o código para aceitar subgrupos retangulares.
6. Quanto maior a grade, maior a entrada da RNA e maior a necessidade de dados.



## 12. Conclusão

O trabalho demonstrou que uma RNA multicamadas pode ser usada para reconhecer padrões de validade em tabuleiros de Sudoku, especialmente no caso 4x4. Nesse cenário, a rede apresentou alta acurácia tanto no treinamento quanto no teste.

A generalização para NxN mostrou que o aumento do tamanho da grade torna o problema significativamente mais difícil. No caso 9x9, a RNA apresentou dificuldade de generalização, evidenciada pela diferença entre a acurácia de treino e a acurácia de teste.

Assim, conclui-se que a RNA pode ser utilizada como uma ferramenta auxiliar, mas a garantia de validade da solução depende das regras formais do Sudoku e do algoritmo de busca com restrições. Essa combinação permite resolver os tabuleiros gerados e, ao mesmo tempo, discutir as limitações da aplicação de redes neurais em problemas de raciocínio lógico.
