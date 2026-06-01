import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = 4
BOX = 2
S = [1, 2, 3, 4]


def is_safe(board, r, c, value):
    if value in board[r, :]:
        return False
    if value in board[:, c]:
        return False
    br = (r // BOX) * BOX
    bc = (c // BOX) * BOX
    if value in board[br:br + BOX, bc:bc + BOX]:
        return False
    return True


def is_valid_complete(board):
    required = set(S)

    if board is None or board.shape != (N, N):
        return False

    if np.any((board < 1) | (board > 4)):
        return False

    for i in range(N):
        if set(board[i, :]) != required:
            return False
        if set(board[:, i]) != required:
            return False

    for br in range(0, N, BOX):
        for bc in range(0, N, BOX):
            subgroup = board[br:br + BOX, bc:bc + BOX].ravel()
            if set(subgroup) != required:
                return False

    return True


def generate_all_solutions():
    board = np.zeros((N, N), dtype=int)
    solutions = []

    def backtrack(pos=0):
        if pos == N * N:
            solutions.append(board.copy())
            return

        r, c = divmod(pos, N)

        for value in S:
            if is_safe(board, r, c, value):
                board[r, c] = value
                backtrack(pos + 1)
                board[r, c] = 0

    backtrack()
    return solutions


def one_hot(board):
    x = np.zeros((N, N, len(S)), dtype=float)

    for r in range(N):
        for c in range(N):
            value = board[r, c]
            if value in S:
                x[r, c, value - 1] = 1.0

    return x.reshape(1, -1)


def corrupt_board(board, rng):
    for _ in range(100):
        corrupted = board.copy()
        operation = rng.choice([
            "row_duplicate",
            "col_duplicate",
            "box_duplicate",
            "random_grid",
            "swap_cells"
        ])

        if operation == "row_duplicate":
            r = rng.integers(0, N)
            c1, c2 = rng.choice(N, 2, replace=False)
            corrupted[r, c2] = corrupted[r, c1]

        elif operation == "col_duplicate":
            c = rng.integers(0, N)
            r1, r2 = rng.choice(N, 2, replace=False)
            corrupted[r2, c] = corrupted[r1, c]

        elif operation == "box_duplicate":
            br = rng.choice([0, 2])
            bc = rng.choice([0, 2])
            cells = [(br + i, bc + j) for i in range(BOX) for j in range(BOX)]
            first, second = rng.choice(len(cells), 2, replace=False)
            r1, c1 = cells[first]
            r2, c2 = cells[second]
            corrupted[r2, c2] = corrupted[r1, c1]

        elif operation == "random_grid":
            corrupted = rng.integers(1, N + 1, size=(N, N))

        else:
            cells = rng.choice(N * N, 2, replace=False)
            r1, c1 = divmod(int(cells[0]), N)
            r2, c2 = divmod(int(cells[1]), N)
            corrupted[r1, c1], corrupted[r2, c2] = corrupted[r2, c2], corrupted[r1, c1]

        if not is_valid_complete(corrupted):
            return corrupted

    raise RuntimeError("Não foi possível gerar uma amostra inválida.")


def create_dataset(solutions, negatives_per_solution=4, seed=42):
    rng = np.random.default_rng(seed)
    X = []
    y = []

    for solution in solutions:
        for _ in range(negatives_per_solution):
            X.append(one_hot(solution).ravel())
            y.append(1)

            invalid = corrupt_board(solution, rng)
            X.append(one_hot(invalid).ravel())
            y.append(0)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, 1)
    indexes = rng.permutation(len(X))

    return X[indexes], y[indexes]


@dataclass
class MLP:
    input_size: int
    hidden_sizes: tuple = (64, 32)
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
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

    def forward(self, X):
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

    def fit(self, X, y, epochs=1000, batch_size=64):
        rng = np.random.default_rng(self.seed)
        n = len(X)

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

            if epoch % 200 == 0:
                print(
                    f"Época {epoch:4d} | "
                    f"loss={self.loss(X, y):.4f} | "
                    f"acc={self.accuracy(X, y):.4f}"
                )

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def loss(self, X, y):
        yhat = self.predict_proba(X)
        eps = 1e-9
        return float(-np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps)))

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_ratio))
    train_indexes = indexes[:cut]
    test_indexes = indexes[cut:]

    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]


def candidates(board, r, c):
    return [value for value in S if is_safe(board, r, c, value)]


def choose_empty_cell(board):
    best_cell = None
    best_candidates = None

    for r in range(N):
        for c in range(N):
            if board[r, c] == 0:
                current_candidates = candidates(board, r, c)

                if best_cell is None or len(current_candidates) < len(best_candidates):
                    best_cell = (r, c)
                    best_candidates = current_candidates

                if len(best_candidates) == 0:
                    return best_cell, best_candidates

    return best_cell, best_candidates


def solve_with_rules_and_rna(puzzle, model):
    board = puzzle.copy()

    def backtrack():
        cell, current_candidates = choose_empty_cell(board)

        if cell is None:
            probability = float(model.predict_proba(one_hot(board))[0, 0])
            if is_valid_complete(board) and probability >= 0.5:
                return board.copy()
            return None

        if len(current_candidates) == 0:
            return None

        r, c = cell
        scored_candidates = []

        for value in current_candidates:
            board[r, c] = value
            score = float(model.predict_proba(one_hot(board))[0, 0])
            scored_candidates.append((score, value))
            board[r, c] = 0

        for _, value in sorted(scored_candidates, reverse=True):
            board[r, c] = value
            result = backtrack()

            if result is not None:
                return result

            board[r, c] = 0

        return None

    return backtrack()


def generate_random_puzzle(solutions, blanks=8, seed=7):
    rng = np.random.default_rng(seed)
    solution = solutions[rng.integers(0, len(solutions))].copy()
    puzzle = solution.copy()
    positions = rng.choice(N * N, size=blanks, replace=False)

    for position in positions:
        r, c = divmod(int(position), N)
        puzzle[r, c] = 0

    return puzzle, solution


def plot_board(board, title, filename):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(N + 1):
        linewidth = 2.2 if i % BOX == 0 else 0.8
        ax.plot([0, N], [i, i], linewidth=linewidth)
        ax.plot([i, i], [0, N], linewidth=linewidth)

    for r in range(N):
        for c in range(N):
            value = board[r, c]
            if value != 0:
                ax.text(c + 0.5, N - r - 0.5, str(value), ha="center", va="center", fontsize=18)

    ax.set_title(title)
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    solutions = generate_all_solutions()
    print(f"Soluções completas válidas 4x4 geradas: {len(solutions)}")

    X, y = create_dataset(solutions, negatives_per_solution=4, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)

    print(f"Amostras de treino: {X_train.shape[0]}")
    print(f"Amostras de teste : {X_test.shape[0]}")

    model = MLP(input_size=X.shape[1], hidden_sizes=(64, 32), lr=0.03, seed=42)
    model.fit(X_train, y_train, epochs=1000, batch_size=64)

    print(f"\nAcurácia treino: {model.accuracy(X_train, y_train):.4f}")
    print(f"Acurácia teste : {model.accuracy(X_test, y_test):.4f}")

    os.makedirs("imagens", exist_ok=True)

    for example_id, seed in enumerate([9, 14, 31], start=1):
        puzzle, original_solution = generate_random_puzzle(solutions, blanks=8, seed=seed)
        solved = solve_with_rules_and_rna(puzzle, model)
        probability = float(model.predict_proba(one_hot(solved))[0, 0]) if solved is not None else 0.0

        print(f"\n========== Exemplo {example_id} ==========");
        print("Tabuleiro inicial aleatório:")
        print(puzzle)
        print("\nSolução base usada para gerar o puzzle:")
        print(original_solution)
        print("\nSolução final encontrada:")
        print(solved)
        print("\nVálido pelas regras:", is_valid_complete(solved))
        print("Probabilidade de validade pela RNA:", probability)

        plot_board(puzzle, f"Sudoku 4x4 inicial {example_id}", f"imagens/sudoku_inicial_{example_id}.png")
        plot_board(solved, f"Sudoku 4x4 resolvido {example_id}", f"imagens/sudoku_resolvido_{example_id}.png")

    print("\nImagens geradas na pasta: imagens/")


if __name__ == "__main__":
    main()
