
# Sudoku com Rede Neural Artificial (RNA)


Alunos:  
- Gabriel Yuri Cavalcante De Castro - 22350996
- Marcele Azevedo de Paula Oliveira-22353160

## Descrição

Este projeto apresenta uma solução para o problema do Sudoku utilizando uma Rede Neural Artificial (RNA) multicamadas em Python.

A proposta foi dividida em duas etapas:

* Implementação e treinamento para o Sudoku 4x4.
* Generalização da solução para grades NxN (9x9, 16x16 e outras grades compatíveis).

A RNA é utilizada para reconhecer padrões de validade em tabuleiros de Sudoku, enquanto a resolução do quebra-cabeça é realizada por um algoritmo baseado em restrições e busca.

---

## Objetivos

* Gerar soluções válidas de Sudoku.
* Criar conjuntos de treinamento e teste.
* Treinar uma RNA multicamadas para reconhecer tabuleiros válidos e inválidos.
* Gerar tabuleiros incompletos aleatórios.
* Resolver automaticamente os tabuleiros gerados.
* Discutir as dificuldades da generalização de Sudoku 4x4 para Sudoku NxN.

---

## Estrutura do Projeto

```text
sudoku-rna/
├── README.md
├── relatorio.md
├── requirements.txt
├── src/
│   ├── sudoku4_rna.py
│   └── sudoku_n_rna.py
└── imagens/
    ├── imagens4x4/
    └── imagensNxN/
```


## Requisitos

Python 3.10 ou superior.

Instale as dependências utilizando:

```bash
pip install -r requirements.txt
```

ou

```bash
pip install numpy matplotlib scikit-learn
```

---

## Sudoku 4x4

O Sudoku 4x4 utiliza:

```text
Símbolos: {1, 2, 3, 4}
Subgrupos: 2x2
```

Regras:

* Cada linha contém os números 1 a 4 sem repetição.
* Cada coluna contém os números 1 a 4 sem repetição.
* Cada subgrupo 2x2 contém os números 1 a 4 sem repetição.

### Execução

```bash
python src/sudoku4_rna.py
```

### Exemplo de saída

```text
Tabuleiro inicial:
[[2 0 1 0]
 [1 3 2 0]
 [3 0 0 0]
 [0 0 3 2]]

Solução encontrada:
[[2 4 1 3]
 [1 3 2 4]
 [3 2 4 1]
 [4 1 3 2]]
```

---

## Sudoku NxN

A versão generalizada permite executar grades maiores.

### Sudoku 9x9

```bash
python src/sudoku_n_rna.py --n 9 --solucoes 200 --epocas 80 --vazios 20
```

### Sudoku 16x16

```bash
python src/sudoku_n_rna.py --n 16
```

---

adas.
