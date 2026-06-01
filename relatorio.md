# Relatório – Resolução de Sudoku NxN com Rede Neural Artificial Multicamadas

## Disciplina

Inteligência Artificial

## Professor

Edjard Mota

## Integrantes

- Gabriel Yuri Cavalcante de Castro – 22350996
- Marcele Azevedo de Paula Oliveira – 22353160

## Universidade

Universidade Federal do Amazonas (UFAM)

---

# 1. Introdução

A Inteligência Artificial (IA) possui como um de seus objetivos o desenvolvimento de sistemas capazes de aprender, raciocinar e tomar decisões em problemas complexos. Entre os diversos problemas estudados na área, o Sudoku destaca-se por exigir simultaneamente aprendizado, busca e satisfação de restrições.

O Sudoku é tradicionalmente conhecido como um jogo de lógica, porém pode ser formalizado como um Problema de Satisfação de Restrições (Constraint Satisfaction Problem – CSP). Nesse contexto, cada célula da grade representa uma variável que deve assumir um valor válido sem violar as restrições impostas pelas linhas, colunas e blocos.

O objetivo deste trabalho foi desenvolver uma solução baseada em Redes Neurais Artificiais Multicamadas (MLP) capaz de resolver instâncias de Sudoku de diferentes dimensões. A implementação foi construída de forma genérica para suportar grades 4x4, 9x9 e 16x16, demonstrando a possibilidade de generalização da abordagem para diferentes tamanhos de problema.

Além do aprendizado realizado pela rede neural, foi incorporado um mecanismo de validação simbólica baseado nas regras do Sudoku. Dessa forma, a solução combina aprendizado estatístico com raciocínio lógico, aproximando-se dos conceitos modernos de IA neurossimbólica.

---

# 2. Fundamentação Teórica

## 2.1 Sudoku como Problema de Satisfação de Restrições

O Sudoku pode ser representado como um CSP, no qual:

- As variáveis correspondem às células da grade.
- O domínio de cada variável corresponde aos valores possíveis.
- As restrições determinam quais combinações são válidas.

Uma solução válida deve satisfazer simultaneamente:

- Cada célula contém exatamente um valor.
- Não há repetição em linhas.
- Não há repetição em colunas.
- Não há repetição dentro dos blocos.

Portanto, resolver um Sudoku significa encontrar uma atribuição de valores capaz de satisfazer todas essas restrições ao mesmo tempo.

---

## 2.2 Sudoku e Raciocínio Lógico

Embora possa ser tratado como um problema de preenchimento de padrões, o Sudoku exige raciocínio baseado em restrições.

Para determinar corretamente o valor de uma única célula é necessário considerar simultaneamente:

- os valores já presentes na linha;
- os valores presentes na coluna;
- os valores presentes no bloco correspondente.

Essa característica faz com que o Sudoku seja frequentemente utilizado em pesquisas relacionadas à representação de conhecimento, lógica proposicional, SAT (Satisfatibilidade Booleana) e sistemas de inferência.

---

## 2.3 Redes Neurais Artificiais Multicamadas

As Redes Neurais Artificiais Multicamadas (MLP – Multilayer Perceptron) são modelos de aprendizado supervisionado compostos por:

- camada de entrada;
- camadas ocultas;
- camada de saída.

Durante o treinamento, os pesos da rede são ajustados utilizando retropropagação do erro (backpropagation), permitindo que a rede aprenda padrões presentes nos dados.

Neste trabalho, a RNA foi treinada para aprender a relação entre um tabuleiro incompleto e sua respectiva solução completa.

---

## 2.4 IA Neurosimbólica

Uma limitação importante das redes neurais é que elas aprendem padrões estatísticos, mas não possuem conhecimento explícito das regras do domínio.

No Sudoku, isso significa que uma RNA pode prever números repetidos em uma linha ou coluna mesmo quando tais valores são proibidos pelas regras do problema.

Para superar essa limitação, foi utilizada uma abordagem híbrida, combinando:

- aprendizado neural;
- validação simbólica;
- backtracking.

Essa integração caracteriza uma abordagem neurossimbólica, na qual métodos estatísticos e simbólicos trabalham de forma complementar.

---

# 3. Representação do Problema

A implementação foi desenvolvida de forma genérica para suportar Sudoku NxN.

Os experimentos contemplam:

| Dimensão | Bloco |
|-----------|---------|
| 4x4 | 2x2 |
| 9x9 | 3x3 |
| 16x16 | 4x4 |

O tamanho do bloco é calculado automaticamente pela raiz quadrada da dimensão da grade.

Cada célula pode assumir valores pertencentes ao conjunto:

S = {1, 2, ..., N}

onde N representa a dimensão do Sudoku.

As células vazias são representadas pelo valor 0.

Para alimentar a RNA, foi utilizada codificação One-Hot Encoding.

Assim, cada célula pode assumir:

- estado vazio;
- valor 1;
- valor 2;
- ...
- valor N.

Consequentemente, cada posição da grade é representada por N+1 atributos.

---

# 4. Geração dos Dados

O conjunto de treinamento foi gerado artificialmente.

Inicialmente são construídas grades completas válidas de Sudoku. Posteriormente, algumas posições são removidas aleatoriamente para produzir tabuleiros incompletos.

Cada amostra possui:

### Entrada

Tabuleiro incompleto.

### Saída

Tabuleiro completo correspondente.

Essa estratégia garante que todas as soluções utilizadas durante o treinamento sejam válidas.

Além disso, a remoção aleatória das pistas produz diferentes níveis de dificuldade e aumenta a diversidade dos exemplos.

---

# 5. Arquitetura da Rede Neural

A arquitetura implementada é composta por:

```text
Camada de Entrada
↓
Camada Oculta 1 (ReLU)
↓
Camada Oculta 2 (ReLU)
↓
Camada Oculta 3 (ReLU)
↓
Camada de Saída
```

O número de neurônios da entrada e saída é calculado automaticamente de acordo com o tamanho da grade.

Dessa forma, a mesma implementação pode ser utilizada para Sudoku 4x4, 9x9 ou 16x16.

A função de perda utilizada foi Cross Entropy Loss, adequada para problemas de classificação multiclasse.

O treinamento foi realizado utilizando o algoritmo Adam.

---

# 6. Funcionamento da Solução

A solução desenvolvida é executada em quatro etapas principais.

### Etapa 1 – Geração dos Dados

São produzidos tabuleiros válidos de Sudoku e versões incompletas desses tabuleiros.

### Etapa 2 – Treinamento

A RNA recebe tabuleiros incompletos como entrada e aprende a prever as soluções completas.

### Etapa 3 – Predição

Após o treinamento, a rede gera uma solução para um novo tabuleiro.

### Etapa 4 – Validação Simbólica

A solução produzida é analisada por um mecanismo simbólico responsável por verificar o cumprimento das regras do Sudoku.

Quando necessário, um algoritmo de backtracking guiado pelas probabilidades produzidas pela RNA corrige a solução.

---

# 7. Validação Simbólica

A validação simbólica verifica:

- linhas;
- colunas;
- blocos.

Uma solução somente é considerada válida quando:

- todas as linhas contêm os valores permitidos sem repetição;
- todas as colunas contêm os valores permitidos sem repetição;
- todos os blocos contêm os valores permitidos sem repetição.

Essa etapa garante consistência lógica ao resultado final.

---

# 8. Resultados Experimentais

Os testes demonstraram que a RNA é capaz de aprender padrões de preenchimento em diferentes tamanhos de Sudoku.

Durante o treinamento observou-se:

- redução progressiva da função de perda;
- aumento da acurácia;
- melhoria da qualidade das previsões.

Em diversos casos a rede produziu soluções próximas da resposta correta. Entretanto, algumas previsões ainda apresentaram violações das regras do Sudoku.

Nesses casos, o mecanismo de validação simbólica e o backtracking foram responsáveis por produzir uma solução final válida.

---

# 9. Dificuldade de Geração de Amostras

Um dos principais desafios do problema está relacionado à explosão combinatória.

À medida que o tamanho do Sudoku aumenta:

- cresce o número de células;
- cresce o número de combinações possíveis;
- cresce o espaço de busca.

Consequentemente, torna-se cada vez mais difícil gerar exemplos suficientes para representar adequadamente todas as possibilidades do problema.

Esse crescimento evidencia que o Sudoku não pode ser tratado apenas como uma tarefa de classificação, mas também como um problema de raciocínio e satisfação de restrições.

---

# 10. Generalização para Sudoku NxN

Uma das principais contribuições da implementação desenvolvida foi sua capacidade de generalização.

O sistema foi projetado para adaptar automaticamente:

- tamanho da grade;
- tamanho dos blocos;
- quantidade de símbolos;
- dimensões da entrada;
- dimensões da saída.

Assim, o mesmo código pode ser utilizado para Sudoku 4x4, 9x9 e 16x16.

Entretanto, o aumento da dimensão provoca crescimento significativo do custo computacional, especialmente durante a etapa de busca simbólica.

---

# 11. Relação com os Conteúdos da Disciplina

Este projeto envolve diversos conceitos estudados ao longo da disciplina:

- Representação de Conhecimento;
- Lógica Proposicional;
- SAT;
- CSP;
- Heurísticas para Sudoku;
- Redes Neurais Artificiais;
- Aprendizado Supervisionado;
- IA Neurosimbólica.

O trabalho demonstra que problemas de raciocínio continuam exigindo mecanismos simbólicos mesmo quando técnicas modernas de aprendizado de máquina são utilizadas.

---

# 12. Participação da Equipe

O trabalho foi desenvolvido em equipe, contemplando:

- estudo dos materiais da disciplina;
- implementação da solução;
- realização dos experimentos;
- análise dos resultados;
- elaboração da documentação.

Equipe:

- Gabriel Yuri Cavalcante de Castro – 22350996
- Marcele Azevedo de Paula Oliveira – 22353160

---

# 13. Conclusão

Os resultados obtidos demonstraram que Redes Neurais Artificiais Multicamadas podem aprender padrões de preenchimento em Sudoku e produzir soluções consistentes para diferentes tamanhos de grade.

Entretanto, verificou-se que a RNA, isoladamente, não garante o cumprimento de todas as restrições do problema.

Por esse motivo, a utilização conjunta de validação simbólica e backtracking mostrou-se fundamental para garantir a correção lógica das soluções.

A integração entre aprendizado estatístico e raciocínio simbólico permitiu construir uma solução mais robusta, evidenciando na prática os conceitos de IA neurossimbólica estudados na disciplina.

Por fim, a generalização da implementação para Sudoku 4x4, 9x9 e 16x16 demonstrou a flexibilidade da abordagem proposta e reforçou a importância da combinação entre aprendizado de máquina e representação simbólica na resolução de problemas complexos de Inteligência Artificial.
