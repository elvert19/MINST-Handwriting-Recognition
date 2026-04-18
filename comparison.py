"""
MNIST: Rust Neural Engine vs Pure Python (truly no NumPy in training)
=====================================================================
NumPy is only used for:
  - Loading the MNIST binary files (IO parsing, not math)
  - Final accuracy evaluation (argmax over predictions)

All training math — matrix multiply, relu, softmax, Adam, backprop —
is plain Python floats and lists. No C, no BLAS, no vectorisation.

⚠ Pure Python matrix multiply is ~500x slower than NumPy.
  Full 60k dataset would take hours. We use N_SAMPLES = 5000.
  Reduce EPOCHS or N_SAMPLES further if it's too slow on your machine.
"""

import neural_engine
import numpy as np
import math, random, gzip, os, time
import requests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── tuneable ─────────────────────────────────────────────────────────────────
N_SAMPLES  = 5_000   # samples used by BOTH models — increase if patient
EPOCHS     = 10
BATCH_SIZE = 64

# ── MNIST loader (numpy for IO only) ─────────────────────────────────────────
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for name, filename in FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"  Downloading {filename}...")
            r = requests.get(BASE_URL + filename)
            with open(path, "wb") as f:
                f.write(r.content)

def load_images(path):
    with gzip.open(path, "rb") as f:
        f.read(16)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float64) / 255.0

def load_labels(path):
    with gzip.open(path, "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot_np(labels, n=10):
    """NumPy one-hot — used for Rust engine input."""
    out = np.zeros((len(labels), n), dtype=np.float64)
    out[np.arange(len(labels)), labels] = 1.0
    return out

# ── Pure Python matrix helpers ────────────────────────────────────────────────
# All operations below use only Python built-ins and math module.

def zeros2d(r, c):
    return [[0.0] * c for _ in range(r)]

def mat_add_bias(Z, b):
    """Add bias row (list of floats) to every row of Z."""
    nc = len(b)
    return [[Z[i][j] + b[j] for j in range(nc)] for i in range(len(Z))]

def mat_mul(A, B):
    """Pure Python matrix multiply — O(n³) with no BLAS."""
    nr, inner, nc = len(A), len(B), len(B[0])
    C = zeros2d(nr, nc)
    for i in range(nr):
        Ai = A[i]
        Ci = C[i]
        for k in range(inner):
            aik = Ai[k]
            if aik == 0.0:
                continue  # skip exact zeros (helps post-ReLU layers)
            Bk = B[k]
            for j in range(nc):
                Ci[j] += aik * Bk[j]
    return C

def mat_T(A):
    """Transpose."""
    nr, nc = len(A), len(A[0])
    return [[A[i][j] for i in range(nr)] for j in range(nc)]

def mat_elem_mul(A, B):
    """Element-wise multiply."""
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def col_sum(A):
    """Sum rows → 1-D list (like axis=0 sum)."""
    nc = len(A[0])
    return [sum(A[i][j] for i in range(len(A))) for j in range(nc)]

def relu_f(Z):
    return [[x if x > 0.0 else 0.0 for x in row] for row in Z]

def relu_d(Z):
    return [[1.0 if x > 0.0 else 0.0 for x in row] for row in Z]

def softmax_f(Z):
    out = []
    for row in Z:
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = sum(exps)
        out.append([v / s for v in exps])
    return out

def cce_loss(pred, true):
    total = 0.0
    for p_row, t_row in zip(pred, true):
        for p, t in zip(p_row, t_row):
            if t > 0.0:
                total -= t * math.log(p + 1e-9)
    return total / len(pred)

def randn():
    """Box-Muller normal sample — no numpy."""
    u1, u2 = random.random(), random.random()
    return math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2.0 * math.pi * u2)

def he_weights(fan_in, fan_out):
    scale = math.sqrt(2.0 / fan_in)
    return [[randn() * scale for _ in range(fan_out)] for _ in range(fan_in)]

def arr_sub(a, b):    return [x - y for x, y in zip(a, b)]
def arr_mul(a, s):    return [x * s for x in a]
def arr_add(a, b):    return [x + y for x, y in zip(a, b)]
def arr_sq(a):        return [x * x for x in a]
def arr_sqrt_eps(a, e=1e-8): return [math.sqrt(x) + e for x in a]
def arr_div(a, b):    return [x / y for x, y in zip(a, b)]

# ── Pure Python Adam optimizer ────────────────────────────────────────────────

class PureAdam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, param, grad, key):
        """
        param and grad are 2-D lists (rows × cols).
        Returns updated param as a new 2-D list.
        """
        nr, nc = len(param), len(param[0])
        flat_p = [param[i][j] for i in range(nr) for j in range(nc)]
        flat_g = [grad[i][j]  for i in range(nr) for j in range(nc)]

        if key not in self.m:
            self.m[key] = [0.0] * len(flat_p)
            self.v[key] = [0.0] * len(flat_p)
        self.t += 1

        b1t = self.b1 ** self.t
        b2t = self.b2 ** self.t
        new_m, new_v, new_p = [], [], []

        for p, g, m, v in zip(flat_p, flat_g, self.m[key], self.v[key]):
            m_new = self.b1 * m + (1 - self.b1) * g
            v_new = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m_new / (1 - b1t)
            v_hat = v_new / (1 - b2t)
            new_m.append(m_new)
            new_v.append(v_new)
            new_p.append(p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))

        self.m[key] = new_m
        self.v[key] = new_v
        return [[new_p[i * nc + j] for j in range(nc)] for i in range(nr)]

    def step_1d(self, param, grad, key):
        """Same but for 1-D bias lists."""
        n = len(param)
        if key not in self.m:
            self.m[key] = [0.0] * n
            self.v[key] = [0.0] * n
        self.t += 1

        b1t = self.b1 ** self.t
        b2t = self.b2 ** self.t
        new_m, new_v, new_p = [], [], []

        for p, g, m, v in zip(param, grad, self.m[key], self.v[key]):
            m_new = self.b1 * m + (1 - self.b1) * g
            v_new = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m_new / (1 - b1t)
            v_hat = v_new / (1 - b2t)
            new_m.append(m_new)
            new_v.append(v_new)
            new_p.append(p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))

        self.m[key] = new_m
        self.v[key] = new_v
        return new_p


# ── Pure Python network (784 → 128 → 64 → 10) ────────────────────────────────

class PurePythonNet:
    """
    All weights are plain Python lists of lists.
    All operations are Python loops — no C, no BLAS.
    Architecture matches the Rust model exactly.
    """
    def __init__(self):
        self.W1 = he_weights(784, 128); self.b1 = [0.0] * 128
        self.W2 = he_weights(128, 64);  self.b2 = [0.0] * 64
        self.W3 = he_weights(64,  10);  self.b3 = [0.0] * 10
        self.opt = PureAdam(lr=0.001)
        self.cache = {}

    def forward(self, X):
        """X is a list of lists (batch × 784)."""
        Z1 = mat_add_bias(mat_mul(X, self.W1), self.b1)
        A1 = relu_f(Z1)
        Z2 = mat_add_bias(mat_mul(A1, self.W2), self.b2)
        A2 = relu_f(Z2)
        Z3 = mat_add_bias(mat_mul(A2, self.W3), self.b3)
        A3 = softmax_f(Z3)
        self.cache = dict(X=X, Z1=Z1, A1=A1, Z2=Z2, A2=A2, A3=A3)
        return A3

    def backward(self, y_true):
        """y_true is a list of lists (batch × 10, one-hot)."""
        n = len(y_true)
        c = self.cache

        # output layer gradient (CCE + Softmax combined)
        inv_n = 1.0 / n
        dZ3 = [[(c["A3"][i][j] - y_true[i][j]) * inv_n
                for j in range(10)] for i in range(n)]

        dW3 = mat_mul(mat_T(c["A2"]), dZ3)
        db3 = col_sum(dZ3)

        dA2 = mat_mul(dZ3, mat_T(self.W3))
        dZ2 = mat_elem_mul(dA2, relu_d(c["Z2"]))
        dW2 = mat_mul(mat_T(c["A1"]), dZ2)
        db2 = col_sum(dZ2)

        dA1 = mat_mul(dZ2, mat_T(self.W2))
        dZ1 = mat_elem_mul(dA1, relu_d(c["Z1"]))
        dW1 = mat_mul(mat_T(c["X"]), dZ1)
        db1 = col_sum(dZ1)

        self.W1 = self.opt.step(self.W1, dW1, "W1")
        self.b1 = self.opt.step_1d(self.b1, db1, "b1")
        self.W2 = self.opt.step(self.W2, dW2, "W2")
        self.b2 = self.opt.step_1d(self.b2, db2, "b2")
        self.W3 = self.opt.step(self.W3, dW3, "W3")
        self.b3 = self.opt.step_1d(self.b3, db3, "b3")

    def predict(self, X):
        return self.forward(X)


# ── Training loops ────────────────────────────────────────────────────────────

def np_to_lists(arr):
    """Convert numpy array to plain Python list of lists."""
    return arr.tolist()

def train_python(X_train_np, y_train_np, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train the pure Python model. X and y are converted from numpy to lists."""
    print("  Converting data to plain Python lists...")
    X = np_to_lists(X_train_np)
    y = np_to_lists(y_train_np)
    n = len(X)

    model = PurePythonNet()
    losses, times = [], []

    for epoch in range(epochs):
        t0 = time.time()
        idx = list(range(n))
        random.shuffle(idx)
        X_s = [X[i] for i in idx]
        y_s = [y[i] for i in idx]

        epoch_loss, batches = 0.0, 0
        for s in range(0, n, batch_size):
            Xb = X_s[s:s + batch_size]
            yb = y_s[s:s + batch_size]
            out = model.forward(Xb)
            epoch_loss += cce_loss(out, yb)
            model.backward(yb)
            batches += 1

        losses.append(epoch_loss / batches)
        times.append(time.time() - t0)
        print(f"  [Python] Epoch {epoch+1:>2}/{epochs}  "
              f"loss={losses[-1]:.4f}  ({times[-1]:.2f}s)")

    return model, losses, times


def eval_python(model, X_test_np, y_test_np):
    """Evaluate. Runs in pure Python — may take a moment."""
    X = np_to_lists(X_test_np)
    preds = model.predict(X)
    correct = sum(
        1 for p, t in zip(preds, y_test_np.tolist())
        if preds[preds.index(p)].index(max(p)) == int(t)
    )
    # cleaner version:
    correct = sum(
        1 for i, row in enumerate(preds)
        if row.index(max(row)) == int(y_test_np[i])
    )
    return correct / len(preds) * 100


def train_rust(X_train_np, y_train_ohe_np, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model = neural_engine.Sequential()
    model.add_dense(784, 128); model.add_relu()
    model.add_dense(128, 64);  model.add_relu()
    model.add_dense(64, 10);   model.add_softmax()
    model.set_optimizer("adam", 0.001)
    model.set_loss("cce")

    losses, times = [], []
    for epoch in range(epochs):
        t0 = time.time()
        h = model.train(X_train_np, y_train_ohe_np, epochs=1, batch_size=batch_size)
        t = time.time() - t0
        l = float(np.mean(h))
        losses.append(l)
        times.append(t)
        print(f"  [Rust]   Epoch {epoch+1:>2}/{epochs}  loss={l:.4f}  ({t:.2f}s)")

    return model, losses, times


def eval_rust(model, X_test_np, y_test_np):
    preds = np.argmax(np.array(model.predict(X_test_np)), axis=1)
    return np.mean(preds == y_test_np) * 100


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("   MNIST: Rust Engine  vs  Pure Python (no NumPy in training)")
print("=" * 60)
print(f"   Samples : {N_SAMPLES}  |  Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}")
print(f"   ⚠ Pure Python mat-mul is ~500x slower than NumPy.")
print(f"   Reduce N_SAMPLES/EPOCHS at the top of the file if needed.")
print("=" * 60)

download_mnist()
X_all   = load_images("data/train-images-idx3-ubyte.gz")[:N_SAMPLES]
X_test  = load_images("data/t10k-images-idx3-ubyte.gz")
y_all   = load_labels("data/train-labels-idx1-ubyte.gz")[:N_SAMPLES]
y_test  = load_labels("data/t10k-labels-idx1-ubyte.gz")
y_ohe   = one_hot_np(y_all)

# Use same subset for both models — identical data, fair race
print(f"\n🦀 Training Rust Engine ({EPOCHS} epochs, {N_SAMPLES} samples)...")
rust_model, rust_losses, rust_times = train_rust(X_all, y_ohe, EPOCHS, BATCH_SIZE)
rust_acc = eval_rust(rust_model, X_test, y_test)

print(f"\n🐍 Training Pure Python ({EPOCHS} epochs, {N_SAMPLES} samples)...")
py_model, py_losses, py_times = train_python(X_all, y_ohe, EPOCHS, BATCH_SIZE)

print(f"\n  Evaluating pure Python on test set (pure Python predict, slow)...")
py_acc = eval_python(py_model, X_test[:2000], y_test[:2000])  # subset for eval speed

rust_total = sum(rust_times)
py_total   = sum(py_times)
speedup    = py_total / rust_total

print(f"\n{'='*60}")
print(f"  🦀 Rust   — Acc: {rust_acc:.2f}%   Total: {rust_total:.1f}s")
print(f"  🐍 Python — Acc: {py_acc:.2f}%   Total: {py_total:.1f}s")
print(f"   Speedup: {speedup:.1f}× faster with Rust")
print(f"{'='*60}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG, PAN, GRID = "#0D0F14", "#141720", "#1E2130"
RC, PC, TXT   = "#FF6B35", "#4B8BBE", "#E8EAF0"

plt.rcParams.update({
    "font.family": "monospace", "text.color": TXT,
    "axes.facecolor": PAN, "figure.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT,
    "grid.color": GRID, "grid.linestyle": "--", "grid.alpha": 0.4,
})

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"Rust Neural Engine  vs  Pure Python (no NumPy)  ·  MNIST  [{N_SAMPLES} samples]",
    fontsize=13, fontweight="bold", color="white", fontfamily="monospace", y=0.97
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.96, top=0.90, bottom=0.08)
ep = list(range(1, EPOCHS + 1))

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(ep, rust_losses, color=RC, lw=2.5, label=f"🦀 Rust   (final {rust_losses[-1]:.4f})")
ax1.plot(ep, py_losses,   color=PC, lw=2.5, label=f"🐍 Python (final {py_losses[-1]:.4f})", linestyle="--")
ax1.set_title("Training Loss (CCE)", color=TXT, pad=8)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("CCE Loss")
ax1.legend(facecolor=PAN, edgecolor=GRID, labelcolor=TXT)
ax1.grid(True)

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(ep, rust_times, color=RC, lw=2, label="Rust")
ax2.plot(ep, py_times,   color=PC, lw=2, label="Python (pure)", linestyle="--")
ax2.set_title("Time per Epoch (s)", color=TXT, pad=8)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Seconds")
ax2.legend(facecolor=PAN, edgecolor=GRID, labelcolor=TXT, fontsize=8)
ax2.grid(True)

ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(["🦀 Rust", "🐍 Python\n(pure)"], [rust_total, py_total],
               color=[RC, PC], width=0.5)
ax3.set_title("Total Training Time (s)", color=TXT, pad=8)
ax3.set_ylabel("Seconds"); ax3.grid(True, axis="y")
for bar, val in zip(bars, [rust_total, py_total]):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val:.1f}s", ha="center", color="white", fontsize=10, fontweight="bold")

ax4 = fig.add_subplot(gs[1, 1])
speedup_per = [py_times[i] / max(rust_times[i], 1e-9) for i in range(EPOCHS)]
ax4.bar(ep, speedup_per, color="#FFD700", alpha=0.85)
ax4.axhline(1.0, color="white", lw=1, linestyle="--", alpha=0.5)
ax4.set_title("Speedup per Epoch  (×)", color=TXT, pad=8)
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Python time / Rust time")
ax4.grid(True, axis="y")

ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
summary = (
    f"{'─'*30}\n"
    f"  RESULTS SUMMARY\n"
    f"{'─'*30}\n"
    f"  Samples  : {N_SAMPLES}\n"
    f"  Epochs   : {EPOCHS}\n"
    f"  Batch    : {BATCH_SIZE}\n"
    f"{'─'*30}\n"
    f"  🦀 Rust Engine\n"
    f"    Accuracy : {rust_acc:.2f}%\n"
    f"    Total    : {rust_total:.1f}s\n"
    f"    Avg/epoch: {rust_total/EPOCHS:.2f}s\n\n"
    f"  🐍 Pure Python\n"
    f"    Accuracy : {py_acc:.2f}%\n"
    f"    Total    : {py_total:.1f}s\n"
    f"    Avg/epoch: {py_total/EPOCHS:.2f}s\n"
    f"{'─'*30}\n"
    f"   Speedup: {speedup:.1f}×"
)
ax5.text(0.05, 0.97, summary, transform=ax5.transAxes,
         fontsize=9, color=TXT, va="top", family="monospace", linespacing=1.8)

plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("\n  Saved: benchmark_results.png")
plt.show()