import neural_engine
import numpy as np
import requests
import gzip
import os
import matplotlib.pyplot as plt

# ── Download MNIST ───────────────────────────────────────────────────────────

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
            print(f"Downloading {filename}...")
            r = requests.get(BASE_URL + filename)
            with open(path, "wb") as f:
                f.write(r.content)

def load_images(path):
    with gzip.open(path, "rb") as f:
        f.read(16)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)

def load_labels(path):
    with gzip.open(path, "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot(labels, num_classes=10):
    out = np.zeros((len(labels), num_classes), dtype=np.float64)
    out[np.arange(len(labels)), labels] = 1.0
    return out


download_mnist()

X_train = load_images("data/train-images-idx3-ubyte.gz").astype(np.float64) / 255.0
y_train = one_hot(load_labels("data/train-labels-idx1-ubyte.gz"))
X_test  = load_images("data/t10k-images-idx3-ubyte.gz").astype(np.float64)  / 255.0
y_test  = load_labels("data/t10k-labels-idx1-ubyte.gz")

print(f"Train: {X_train.shape} | Test: {X_test.shape}")


model = neural_engine.Sequential()
model.add_dense(784, 128)
model.add_relu()
model.add_dense(128, 64)
model.add_relu()
model.add_dense(64, 10)
model.add_softmax()
model.set_optimizer("adam", 0.001)
model.set_loss("cce")

print(f"Layers: {model.layer_count()}")

print("\nTraining...")
loss_history = model.train(X_train, y_train, epochs=20, batch_size=64)

for i, loss in enumerate(loss_history):
    loss_val = float(np.mean(loss))
    if i % 5 == 0:
        print(f"Epoch {i:>3}: loss = {loss_val:.4f}")


predictions = model.predict(X_test)
predicted_classes = np.argmax(np.array(predictions), axis=1)
accuracy = np.mean(predicted_classes == y_test) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ── Loss curve ───────────────────────────────────────────────────────────────

plt.figure(figsize=(10, 4))
plt.plot([float(np.mean(l)) for l in loss_history], color="#FF6B35", lw=2)
plt.title("Training Loss — Rust Neural Engine")
plt.xlabel("Epoch")
plt.ylabel("CCE Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# ── Sample predictions ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle(f"Sample Predictions  (Accuracy: {accuracy:.2f}%)", fontsize=13)
for i, ax in enumerate(axes.flat):
    img = X_test[i].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    pred = predicted_classes[i]
    true = y_test[i]
    color = "green" if pred == true else "red"
    ax.set_title(f"Pred: {pred} | True: {true}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("predictions.png")
plt.show()

print("\nSaved: loss_curve.png  |  predictions.png")