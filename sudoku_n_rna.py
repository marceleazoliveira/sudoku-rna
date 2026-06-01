import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SudokuConfig:
    n: int = 4
    box_rows: Optional[int] = None
    box_cols: Optional[int] = None

    def __post_init__(self):
        if self.box_rows is None or self.box_cols is None:
            root = int(math.sqrt(self.n))
            if root * root != self.n:
                raise ValueError(
                    "Para usar apenas --n, N precisa ser quadrado perfeito, como 4, 9 ou 16. "
                    "Para outros tamanhos, informe --box-linhas e --box-colunas."
                )
            self.box_rows = root
            self.box_cols = root

        if self.box_rows * self.box_cols != self.n:
            raise ValueError("box_linhas * box_colunas precisa ser igual a N.")

        self.symbols = list(range(1, self.n + 1))


class SudokuRules:
    def __init__(self, config: SudokuConfig):
        self.config = config
        self.n = config.n
        self.box_rows = config.box_rows
        self.box_cols = config.box_cols
        self.symbols = config.symbols
        self.required = set(self.symbols)

    def is_safe(self, board: np.ndarray, row: int, col: int, value: int) -> bool:
        if value in board[row, :]:
            return False
        if value in board[:, col]:
            return False

        br = (row // self.box_rows) * self.box_rows
        bc = (col // self.box_cols) * self.box_cols
        if value in board[br:br + self.box_rows, bc:bc + self.box_cols]:
            return False

        return True

    def _unit_has_no_repetition(self, values: np.ndarray) -> bool:
        values = [int(v) for v in values if int(v) != 0]
        return len(values) == len(set(values))

    def is_valid_partial(self, board: Optional[np.ndarray]) -> bool:
        if board is None or board.shape != (self.n, self.n):
            return False

        if np.any((board < 0) | (board > self.n)):
            return False

        for i in range(self.n):
            if not self._unit_has_no_repetition(board[i, :]):
                return False
            if not self._unit_has_no_repetition(board[:, i]):
                return False

        for br in range(0, self.n, self.box_rows):
            for bc in range(0, self.n, self.box_cols):
                subgroup = board[br:br + self.box_rows, bc:bc + self.box_cols].ravel()
                if not self._unit_has_no_repetition(subgroup):
                    return False

        return True

    def is_valid_complete(self, board: Optional[np.ndarray]) -> bool:
        if not self.is_valid_partial(board):
            return False

        if np.any(board == 0):
            return False

        for i in range(self.n):
            if set(board[i, :]) != self.required:
                return False
            if set(board[:, i]) != self.required:
                return False

        for br in range(0, self.n, self.box_rows):
            for bc in range(0, self.n, self.box_cols):
                subgroup = board[br:br + self.box_rows, bc:bc + self.box_cols].ravel()
                if set(subgroup) != self.required:
                    return False

        return True

    def candidates(self, board: np.ndarray, row: int, col: int) -> List[int]:
        if board[row, col] != 0:
            return []
        return [value for value in self.symbols if self.is_safe(board, row, col, value)]

    def choose_empty_cell(self, board: np.ndarray) -> Tuple[Optional[Tuple[int, int]], List[int]]:
        best_cell = None
        best_candidates = []

        for row in range(self.n):
            for col in range(self.n):
                if board[row, col] == 0:
                    current_candidates = self.candidates(board, row, col)

                    if best_cell is None or len(current_candidates) < len(best_candidates):
                        best_cell = (row, col)
                        best_candidates = current_candidates

                    if len(best_candidates) == 0:
                        return best_cell, best_candidates

        return best_cell, best_candidates


class SudokuGenerator:
    def __init__(self, config: SudokuConfig, seed: int = 42):
        self.config = config
        self.rules = SudokuRules(config)
        self.rng = np.random.default_rng(seed)

    def generate_all_solutions_4x4(self) -> List[np.ndarray]:
        if self.config.n != 4:
            raise ValueError("A geração completa por backtracking foi mantida apenas para o caso 4x4.")

        board = np.zeros((self.config.n, self.config.n), dtype=int)
        solutions = []

        def backtrack(pos: int = 0):
            if pos == self.config.n * self.config.n:
                solutions.append(board.copy())
                return

            row, col = divmod(pos, self.config.n)
            for value in self.config.symbols:
                if self.rules.is_safe(board, row, col, value):
                    board[row, col] = value
                    backtrack(pos + 1)
                    board[row, col] = 0

        backtrack()
        return solutions

    def generate_pattern_solution(self) -> np.ndarray:
        n = self.config.n
        box_rows = self.config.box_rows
        box_cols = self.config.box_cols

        def pattern(row: int, col: int) -> int:
            return (box_cols * (row % box_rows) + row // box_rows + col) % n

        row_bands = self.rng.permutation(n // box_rows)
        rows = [band * box_rows + row for band in row_bands for row in self.rng.permutation(box_rows)]

        col_stacks = self.rng.permutation(n // box_cols)
        cols = [stack * box_cols + col for stack in col_stacks for col in self.rng.permutation(box_cols)]

        symbols = self.rng.permutation(np.arange(1, n + 1))
        board = np.array([[symbols[pattern(row, col)] for col in cols] for row in rows], dtype=int)
        return board

    def generate_solutions(self, amount: int = 600, use_all_4x4: bool = True) -> List[np.ndarray]:
        if self.config.n == 4 and use_all_4x4:
            return self.generate_all_solutions_4x4()

        solutions = []
        seen = set()

        attempts = 0
        max_attempts = amount * 20
        while len(solutions) < amount and attempts < max_attempts:
            attempts += 1
            board = self.generate_pattern_solution()
            key = tuple(board.ravel())
            if key not in seen and self.rules.is_valid_complete(board):
                seen.add(key)
                solutions.append(board)

        if len(solutions) < amount:
            print(f"Aviso: foram geradas {len(solutions)} soluções únicas em vez de {amount}.")

        return solutions

    def make_puzzle(self, solution: np.ndarray, blanks: int) -> np.ndarray:
        n = self.config.n
        puzzle = solution.copy()
        blanks = min(max(blanks, 0), n * n - 1)
        positions = self.rng.choice(n * n, size=blanks, replace=False)

        for position in positions:
            row, col = divmod(int(position), n)
            puzzle[row, col] = 0

        return puzzle

    def corrupt_board(self, board: np.ndarray) -> np.ndarray:
        n = self.config.n

        for _ in range(200):
            corrupted = board.copy()
            operation = self.rng.choice(["row", "col", "box", "random"])

            if operation == "row":
                row = int(self.rng.integers(0, n))
                filled_cols = np.where(corrupted[row, :] != 0)[0]
                if len(filled_cols) == 0:
                    continue
                source_col = int(self.rng.choice(filled_cols))
                target_col = int(self.rng.integers(0, n))
                if target_col == source_col:
                    continue
                corrupted[row, target_col] = corrupted[row, source_col]

            elif operation == "col":
                col = int(self.rng.integers(0, n))
                filled_rows = np.where(corrupted[:, col] != 0)[0]
                if len(filled_rows) == 0:
                    continue
                source_row = int(self.rng.choice(filled_rows))
                target_row = int(self.rng.integers(0, n))
                if target_row == source_row:
                    continue
                corrupted[target_row, col] = corrupted[source_row, col]

            elif operation == "box":
                br = int(self.rng.integers(0, n // self.config.box_rows)) * self.config.box_rows
                bc = int(self.rng.integers(0, n // self.config.box_cols)) * self.config.box_cols
                cells = [
                    (br + i, bc + j)
                    for i in range(self.config.box_rows)
                    for j in range(self.config.box_cols)
                ]
                filled_cells = [(r, c) for r, c in cells if corrupted[r, c] != 0]
                if len(filled_cells) == 0:
                    continue
                source_row, source_col = filled_cells[int(self.rng.integers(0, len(filled_cells)))]
                target_row, target_col = cells[int(self.rng.integers(0, len(cells)))]
                if (source_row, source_col) == (target_row, target_col):
                    continue
                corrupted[target_row, target_col] = corrupted[source_row, source_col]

            else:
                corrupted = self.rng.integers(0, n + 1, size=(n, n))

            if not self.rules.is_valid_partial(corrupted):
                return corrupted

        raise RuntimeError("Não foi possível gerar amostra inválida.")


class DatasetBuilder:
    def __init__(self, config: SudokuConfig, seed: int = 42):
        self.config = config
        self.rules = SudokuRules(config)
        self.generator = SudokuGenerator(config, seed=seed)
        self.rng = np.random.default_rng(seed)

    def encode_board(self, board: np.ndarray) -> np.ndarray:
        n = self.config.n
        encoded = np.zeros((n, n, n + 1), dtype=float)

        for row in range(n):
            for col in range(n):
                value = int(board[row, col])
                if 0 <= value <= n:
                    encoded[row, col, value] = 1.0

        return encoded.reshape(1, -1)

    def create_dataset(
        self,
        solutions: Sequence[np.ndarray],
        partials_per_solution: int = 3,
        negatives_per_positive: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.config.n
        X = []
        y = []

        for solution in solutions:
            positive_boards = [solution]

            for _ in range(partials_per_solution):
                min_blanks = max(1, n)
                max_blanks = max(min_blanks + 1, int(n * n * 0.65))
                blanks = int(self.rng.integers(min_blanks, max_blanks + 1))
                positive_boards.append(self.generator.make_puzzle(solution, blanks=blanks))

            for positive in positive_boards:
                X.append(self.encode_board(positive).ravel())
                y.append(1)

                for _ in range(negatives_per_positive):
                    invalid = self.generator.corrupt_board(positive)
                    X.append(self.encode_board(invalid).ravel())
                    y.append(0)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        indexes = self.rng.permutation(len(X))
        return X[indexes], y[indexes]


@dataclass
class MLP:
    input_size: int
    hidden_sizes: Tuple[int, ...]
    lr: float = 0.03
    seed: int = 42

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        sizes = [self.input_size] + list(self.hidden_sizes) + [1]
        self.W = []
        self.b = []

        for i in range(len(sizes) - 1):
            scale = np.sqrt(2 / sizes[i])
            self.W.append(rng.normal(0, scale, size=(sizes[i], sizes[i + 1])))
            self.b.append(np.zeros((1, sizes[i + 1])))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

    def forward(self, X: np.ndarray):
        A = X
        activations = [X]
        zs = []

        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            A = self.relu(Z)
            zs.append(Z)
            activations.append(A)

        Z = A @ self.W[-1] + self.b[-1]
        A = self.sigmoid(Z)
        zs.append(Z)
        activations.append(A)
        return A, activations, zs

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 600, batch_size: int = 64):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        report_each = max(1, epochs // 5)

        for epoch in range(1, epochs + 1):
            indexes = rng.permutation(n)
            Xs = X[indexes]
            ys = y[indexes]

            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]

                yhat, activations, zs = self.forward(xb)
                delta = (yhat - yb) / len(xb)

                dW = [None] * len(self.W)
                db = [None] * len(self.b)

                dW[-1] = activations[-2].T @ delta
                db[-1] = np.sum(delta, axis=0, keepdims=True)

                for i in reversed(range(len(self.W) - 1)):
                    delta = (delta @ self.W[i + 1].T) * self.relu_grad(zs[i])
                    dW[i] = activations[i].T @ delta
                    db[i] = np.sum(delta, axis=0, keepdims=True)

                for i in range(len(self.W)):
                    self.W[i] -= self.lr * dW[i]
                    self.b[i] -= self.lr * db[i]

            if epoch % report_each == 0 or epoch == 1:
                print(
                    f"Época {epoch:4d} | "
                    f"loss={self.loss(X, y):.4f} | "
                    f"acc={self.accuracy(X, y):.4f}"
                )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        yhat = self.predict_proba(X)
        eps = 1e-9
        return float(-np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class SudokuSolver:
    def __init__(self, config: SudokuConfig, model: MLP, encoder: DatasetBuilder):
        self.config = config
        self.rules = SudokuRules(config)
        self.model = model
        self.encoder = encoder

    def score(self, board: np.ndarray) -> float:
        return float(self.model.predict_proba(self.encoder.encode_board(board))[0, 0])

    def solve(self, puzzle: np.ndarray) -> Optional[np.ndarray]:
        if not self.rules.is_valid_partial(puzzle):
            return None

        board = puzzle.copy()

        def backtrack() -> Optional[np.ndarray]:
            cell, current_candidates = self.rules.choose_empty_cell(board)

            if cell is None:
                if self.rules.is_valid_complete(board):
                    return board.copy()
                return None

            if len(current_candidates) == 0:
                return None

            row, col = cell
            scored_candidates = []

            for value in current_candidates:
                board[row, col] = value
                scored_candidates.append((self.score(board), value))
                board[row, col] = 0

            for _, value in sorted(scored_candidates, reverse=True):
                board[row, col] = value
                result = backtrack()
                if result is not None:
                    return result
                board[row, col] = 0

            return None

        return backtrack()


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_ratio))
    train_indexes = indexes[:cut]
    test_indexes = indexes[cut:]
    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]


def plot_board(config: SudokuConfig, board: np.ndarray, title: str, filename: str):
    n = config.n
    fig_size = max(3.5, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(n + 1):
        horizontal_width = 2.2 if i % config.box_rows == 0 else 0.8
        vertical_width = 2.2 if i % config.box_cols == 0 else 0.8
        ax.plot([0, n], [i, i], linewidth=horizontal_width)
        ax.plot([i, i], [0, n], linewidth=vertical_width)

    fontsize = 18 if n <= 4 else 11
    for row in range(n):
        for col in range(n):
            value = int(board[row, col])
            if value != 0:
                ax.text(col + 0.5, n - row - 0.5, str(value), ha="center", va="center", fontsize=fontsize)

    ax.set_title(title)
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)


def default_hidden_sizes(n: int) -> Tuple[int, int]:
    if n <= 4:
        return 64, 32
    if n <= 9:
        return 162, 81
    return max(256, n * n * 2), max(128, n * n)


def main():
    parser = argparse.ArgumentParser(description="Sudoku NxN com RNA multicamadas e busca por restrições.")
    parser.add_argument("--n", type=int, default=4, help="Tamanho da grade. Exemplos: 4, 9, 16.")
    parser.add_argument("--box-linhas", type=int, default=None, help="Quantidade de linhas do subgrupo.")
    parser.add_argument("--box-colunas", type=int, default=None, help="Quantidade de colunas do subgrupo.")
    parser.add_argument("--solucoes", type=int, default=600, help="Quantidade de soluções amostradas quando N > 4.")
    parser.add_argument("--epocas", type=int, default=None, help="Quantidade de épocas de treinamento.")
    parser.add_argument("--exemplos", type=int, default=3, help="Quantidade de puzzles aleatórios gerados.")
    parser.add_argument("--vazios", type=int, default=None, help="Quantidade de células vazias no puzzle.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    args = parser.parse_args()

    config = SudokuConfig(n=args.n, box_rows=args.box_linhas, box_cols=args.box_colunas)
    generator = SudokuGenerator(config, seed=args.seed)
    dataset_builder = DatasetBuilder(config, seed=args.seed)
    rules = SudokuRules(config)

    use_all_4x4 = config.n == 4
    solutions = generator.generate_solutions(amount=args.solucoes, use_all_4x4=use_all_4x4)

    print(f"Tamanho da grade: {config.n}x{config.n}")
    print(f"Subgrupos: {config.box_rows}x{config.box_cols}")
    print(f"Soluções completas usadas como base: {len(solutions)}")

    X, y = dataset_builder.create_dataset(
        solutions,
        partials_per_solution=4 if config.n <= 4 else 2,
        negatives_per_positive=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=args.seed)

    print(f"Amostras de treino: {X_train.shape[0]}")
    print(f"Amostras de teste : {X_test.shape[0]}")

    epochs = args.epocas
    if epochs is None:
        epochs = 800 if config.n <= 4 else 300

    hidden_sizes = default_hidden_sizes(config.n)
    model = MLP(input_size=X.shape[1], hidden_sizes=hidden_sizes, lr=0.03, seed=args.seed)
    model.fit(X_train, y_train, epochs=epochs, batch_size=64)

    print(f"\nAcurácia treino: {model.accuracy(X_train, y_train):.4f}")
    print(f"Acurácia teste : {model.accuracy(X_test, y_test):.4f}")

    solver = SudokuSolver(config, model, dataset_builder)
    os.makedirs("imagens", exist_ok=True)

    default_blanks = int(config.n * config.n * (0.50 if config.n <= 4 else 0.45))
    blanks = args.vazios if args.vazios is not None else default_blanks

    for example_id in range(1, args.exemplos + 1):
        solution = solutions[int(generator.rng.integers(0, len(solutions)))]
        puzzle = generator.make_puzzle(solution, blanks=blanks)
        solved = solver.solve(puzzle)

        print(f"\n========== Exemplo {example_id} ==========")
        print("Tabuleiro inicial aleatório:")
        print(puzzle)
        print("\nSolução base usada para gerar o puzzle:")
        print(solution)
        print("\nSolução final encontrada:")
        print(solved)
        print("\nTabuleiro inicial consistente pelas regras:", rules.is_valid_partial(puzzle))
        print("Solução válida pelas regras:", rules.is_valid_complete(solved))
        print("Probabilidade de consistência do tabuleiro inicial pela RNA:", solver.score(puzzle))
        if solved is not None:
            print("Probabilidade de validade/consistência da solução pela RNA:", solver.score(solved))

        plot_board(config, puzzle, f"Sudoku {config.n}x{config.n} inicial {example_id}", f"imagens/sudoku_{config.n}x{config.n}_inicial_{example_id}.png")
        if solved is not None:
            plot_board(config, solved, f"Sudoku {config.n}x{config.n} resolvido {example_id}", f"imagens/sudoku_{config.n}x{config.n}_resolvido_{example_id}.png")

    print("\nImagens geradas na pasta: imagens/")


if __name__ == "__main__":
    main()
