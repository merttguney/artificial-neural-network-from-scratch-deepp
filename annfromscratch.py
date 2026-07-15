"""
YAPAY SİNİR AĞI (ANN) - SIFIRDAN NUMPY İLE

Bu kod yapısı, bir sinir ağının temel matematiksel işleyişini gösterir.
Sadece numpy kullanılmıştır.

Mimari: Dinamik katman yapısı
Örnek: [2, 4, 3, 1] → 2 giriş, 4 gizli, 3 gizli, 1 çıkış

AKIŞ:
1. Veri Hazırlama            → Normalizasyon
2. Ağırlık Başlatma          → W ve b parametrelerini oluşturma (dinamik)
3. Aktivasyon Fonksiyonları   → Sigmoid, ReLU, Tanh, Softmax
4. Forward Propagation       → İleri yayılım (N katman, for loop)
5. Loss Fonksiyonları        → Hatayı ölçme (BCE, MSE)
6. Backward Propagation      → Geri yayılım (N katman, dA taşınarak)
7. Gradient Descent          → Parametreleri güncelleme
8. Eğitim Döngüsü            → Tüm adımları birleştir
9. Tahmin ve Accuracy        → Test ve doğruluk hesaplama
"""

import numpy as np


# AŞAMA 1: VERİ HAZIRLAMA


def normalize(X):
    """
    Min-Max Normalization: x' = (x - x_min) / (x_max - x_min)
    Çıkış: [0, 1]
    Neden: Farklı ölçeklerdeki feature'ları dengelemek (gradient descent için)
    """
    x_min = np.min(X, axis=1, keepdims=True)
    x_max = np.max(X, axis=1, keepdims=True)
    return (X - x_min) / (x_max - x_min)



# AŞAMA 2: AĞIRLIK BAŞLATMA


def initialize_weights(layer_sizes):
    """
    Weight Initialization: Küçük rastgele değerlerle başlat

    layer_sizes: [n_input, n_hidden1, n_hidden2, ..., n_output]
    Örnek: [2, 4, 3, 1] → 3 katman ağırlık (W1, W2, W3)

    W ~ N(0, 0.1)  (Gaussian distribution)
    b = 0          (Bias sıfırdan başlar)
    """
    np.random.seed(42)
    params = {}
    L = len(layer_sizes) # kaç katman var aslında onu belirleriz 

    for l in range(1, L): # döngünün amacı aslında mevcut katman ile önceki katman arasında bir ağırlık matrisi oluşturmak
        params[f"W{l}"] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * 0.1 # vanishing gradient riskini azaltmak için
        params[f"b{l}"] = np.zeros((layer_sizes[l], 1))

    return params



# AŞAMA 3: AKTİVASYON FONKSİYONLARI


def sigmoid(z):
    """
    Sigmoid: σ(z) = 1 / (1 + e^(-z))
    Kullanım: Binary classification output layer
    Çıkış: [0, 1]
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Sigmoid Türevi: σ'(z) = σ(z) · (1 - σ(z))
    Backpropagation için
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """
    ReLU: max(0, z)
    Kullanım: Hidden layer'larda (vanishing gradient önler)
    Çıkış: [0, ∞)       
    """
    return np.maximum(0, z) # maximum metodu matrisin her elemanına ayrı ayrı uyguluyor. 


def relu_derivative(z):
    """
    ReLU Türevi: z > 0 ise 1, değilse 0
    Backpropagation için
    """
    return (z > 0).astype(float) # false veya true döndürüyor, 0 ile 1'e dönüştürür.


def tanh(z):
    """
    Tanh: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    Kullanım: Hidden layer (ReLU'ya alternatif)
    Çıkış: [-1, 1]
    """
    return np.tanh(z)


def tanh_derivative(z):
    """
    Tanh Türevi: tanh'(z) = 1 - tanh²(z)
    Backpropagation için
    """
    return 1 - np.tanh(z)**2


def softmax(z):
    """
    Softmax: softmax(z_i) = e^(z_i) / Σ e^(z_j)
    Kullanım: Multi-class classification output layer
    Çıkış: Olasılık dağılımı (toplam = 1)
    """
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)



# AŞAMA 4: FORWARD PROPAGATION (İleri Yayılım)


def forward(X, params):
    """
    FORWARD PROPAGATION: Girdiden çıktıya doğru hesaplama (N katman)

    Her katman için:
      Z[l] = W[l] · A[l-1] + b[l]     (linear transformation)
      A[l] = aktivasyon(Z[l])          (non-linear activation)

    Gizli katmanlar: ReLU
    Çıkış katmanı: Sigmoid

    Return: A_out (tahmin), cache (tüm Z ve A değerleri)
    """
    cache = {"A0": X}  # Girdi verisini sakla(Sıfırıncı katmanın aktivasyonu aslında)
    A = X
    L = len(params) // 2  # Katman sayısı (W ve b çiftleri) => // tam kısmı almak anlamına gelir. !!! 

    # Gizli katmanlar (1'den L-1'e kadar)
    for l in range(1, L):
        Z = np.dot(params[f"W{l}"], A) + params[f"b{l}"] # lineer dönüşüm dediğimiz kısım => ağırlık matrisi ile bir önceki katmanın çıktısını çarp, bias ekle.
        A = relu(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A # her katmanın Z ve A'sını saklıyorum backprop kullanacağım.

    # Çıkış katmanı (L. katman)
    Z = np.dot(params[f"W{L}"], A) + params[f"b{L}"]
    A_out = sigmoid(Z)
    cache[f"Z{L}"] = Z
    cache[f"A{L}"] = A_out

    return A_out, cache # hem tahmini hem de tüm ara değerleri döndürmem lazım



# AŞAMA 5: LOSS FONKSİYONLARI


def compute_loss_bce(Y, A):
    """
    Binary Cross-Entropy Loss:
    L = -(1/m) * Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
    Kullanım: Binary classification
    """
    m = Y.shape[1]
    loss = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return loss


def compute_loss_mse(Y, A):
    """
    Mean Squared Error Loss:
    L = (1/m) * Σ (y - ŷ)²
    Kullanım: Regression
    """
    return np.mean((Y - A)**2)



# AŞAMA 6: BACKWARD PROPAGATION (Geri Yayılım)


def backward(A_out, Y, params, cache):
    """
    BACKWARD PROPAGATION: Chain rule ile gradient hesaplama (N katman)

    Çıkış katmanından geriye doğru her katmanda:
      dZ = dA * aktivasyon_türevi(Z)      (∂L/∂Z)
      dW = (1/m) · dZ · A_prev^T          (∂L/∂W)
      db = (1/m) · Σ dZ                   (∂L/∂b)
      dA_prev = W^T · dZ                  (hatayı bir önceki katmana taşı)

    Return: grads (dW1, db1, dW2, db2, ...)
    """
    grads = {} # tüm gradientleri saklayacak boş bir dictionary
    m = Y.shape[1] 
    L = len(params) // 2

    # Çıkış katmanı (BCE + Sigmoid türevi sadeleşmesi: dZ = A - Y)
    dZ = A_out - Y
    grads[f"dW{L}"] = (1/m) * np.dot(dZ, cache[f"A{L-1}"].T) # dZ ile önceki katmanın aktivasyonunu çarpıyoruz. (1/m) ortalama almaya yarar. 
    grads[f"db{L}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True) # dZ'yi tüm örnekler üzerinde topluyoruz.
    dA = np.dot(params[f"W{L}"].T, dZ) # hatayı bir önceki katmana taşıyoruz dA bir sonraki adımda hidden layerda kullanılacak.

    # Gizli katmanlar (geriye doğru, dA taşınarak)
    for l in reversed(range(1, L)):
        dZ = dA * relu_derivative(cache[f"Z{l}"])
        grads[f"dW{l}"] = (1/m) * np.dot(dZ, cache[f"A{l-1}"].T)
        grads[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(params[f"W{l}"].T, dZ)
# Bu iki kodu birleştirebilirsin, konum olarak L'nin indexinde misin? Eğer ordaysan relu türevi çalıştırma. Gibi bir if else yapısı
    return grads



# AŞAMA 7: GRADIENT DESCENT (Parametre Güncelleme)


def update_params(params, grads, learning_rate):
    """
    GRADIENT DESCENT: Gradient'in tersi yönünde adım at

    Her katman için:
      W[l] = W[l] - η · dW[l]
      b[l] = b[l] - η · db[l]
    """
    L = len(params) // 2

    for l in range(1, L + 1): # l+1'e kadar gidiyor output katmanını da döngüye dahil ediyor tüm katmanların ağırlıklarının güncellenmesi lazım. 
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"] # gradienti learning rate ile çarpıp mevcut ağırlıktan
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return params



# AŞAMA 8: EĞİTİM DÖNGÜSÜ


def train(X, Y, layer_sizes, epochs=10000, learning_rate=0.1):
    """
    EĞİTİM DÖNGÜSÜ: Forward → Loss → Backward → Update

    Her epoch:
      1. Forward propagation  → A_out (tahmin)
      2. Compute loss         → L (hata)
      3. Backward propagation → grads (dW, db)
      4. Gradient descent     → params güncelle

    Return: params, losses
    """
    params = initialize_weights(layer_sizes)
    losses = [] # her epochta loss'u saklayacağız

    for epoch in range(epochs):
        
        A_out, cache = forward(X, params)

        loss = compute_loss_bce(Y, A_out)
        losses.append(loss)

        grads = backward(A_out, Y, params, cache)

        params = update_params(params, grads, learning_rate)

        # Progress
        if epoch % 2000 == 0:  # 2000 epochta bir ekrana yazdır
            print(f"Epoch {epoch}/{epochs} — Loss: {loss:.4f}")

    print(f"Eğitim tamamlandı! Final Loss: {losses[-1]:.4f}")
    return params, losses



# AŞAMA 9: TAHMİN VE ACCURACY


def predict(X, params):
    """
    TAHMİN: Eğitilmiş model ile sınıflandırma

    Forward propagation → A_out (olasılık)
    A_out > 0.5 → Sınıf 1
    A_out ≤ 0.5 → Sınıf 0
    """
    A_out, _ = forward(X, params) # _ => cache'i temsil eder. tahmin aşamasında backward yapmayacağız cache gerek yok.
    predictions = (A_out > 0.5).astype(int) # (A_out > 0.5) true/false üretiyor => .astype(int) bunu 0 ve 1'e çevirir.
    return predictions, A_out


def compute_accuracy(Y, predictions):
    """
    ACCURACY: Doğru tahmin oranı

    Accuracy = (Doğru tahmin sayısı) / (Toplam örnek)
    """
    return np.mean(predictions == Y)



# MAIN: XOR PROBLEMİ


if __name__ == "__main__":
    print("=" * 60)
    print("ANN FROM SCRATCH - XOR PROBLEMİ")
    print("=" * 60)
    print()

    # XOR Problemi
    # XOR doğrusal olarak ayrılamaz → Sinir ağı gerekir
    print("XOR Truth Table:")
    print("  0 XOR 0 = 0")
    print("  0 XOR 1 = 1")
    print("  1 XOR 0 = 1")
    print("  1 XOR 1 = 0")
    print()

    # Veri: Her sütun bir örnek
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])  # (2, 4)

    Y = np.array([[0, 1, 1, 0]])  # (1, 4)

    # Mimari: Input(2) → Hidden(4, ReLU) → Output(1, Sigmoid)
    layer_sizes = [2, 4, 1]

    print(f"Mimari: {layer_sizes}")
    print(f"Giriş boyutu: {X.shape}, Çıkış boyutu: {Y.shape}")
    print("-" * 60)

    # Eğitim
    params, losses = train(
        X, Y,
        layer_sizes=layer_sizes,
        epochs=10000,
        learning_rate=0.5
    )
    print()

    # Tahmin
    print("-" * 60)
    print("SONUÇLAR:")
    predictions, probabilities = predict(X, params)

    for i in range(X.shape[1]):
        print(f"  [{X[0,i]}, {X[1,i]}] → Gerçek: {Y[0,i]} | Tahmin: {predictions[0,i]} (Olasılık: {probabilities[0,i]:.3f})")

    accuracy = compute_accuracy(Y, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)