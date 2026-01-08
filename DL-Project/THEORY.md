# Cơ Sở Lý Thuyết: Deep Neural Networks và Residual Networks

## 1. Deep Neural Networks (DNN)

### 1.1 Giới thiệu

Deep Neural Network (Mạng Neural Sâu) là một lớp mô hình học máy được thiết kế để mô phỏng cách hoạt động của não người. DNN bao gồm nhiều lớp (layers) xử lý thông tin, cho phép học các biểu diễn (representations) phức tạp từ dữ liệu thô.

### 1.2 Kiến trúc cơ bản

#### 1.2.1 Perceptron đơn

Perceptron là đơn vị tính toán cơ bản nhất, mô phỏng một neuron sinh học:

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^T\mathbf{x} + b)$$

Trong đó:
- $\mathbf{x} = [x_1, x_2, ..., x_n]^T$: Vector đầu vào
- $\mathbf{w} = [w_1, w_2, ..., w_n]^T$: Vector trọng số (weights)
- $b$: Độ lệch (bias)
- $f(\cdot)$: Hàm kích hoạt (activation function)
- $y$: Đầu ra

#### 1.2.2 Multi-Layer Perceptron (MLP)

MLP mở rộng perceptron thành nhiều lớp:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Hidden Layer L → Output Layer
```

Với $L$ hidden layers, đầu ra của lớp thứ $l$ được tính:

$$\mathbf{a}^{(l)} = f^{(l)}\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

Trong đó:
- $\mathbf{a}^{(0)} = \mathbf{x}$: Đầu vào
- $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$: Ma trận trọng số lớp $l$
- $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$: Vector bias lớp $l$
- $n_l$: Số neurons trong lớp $l$

### 1.3 Hàm kích hoạt (Activation Functions)

Hàm kích hoạt đưa tính phi tuyến vào mạng, cho phép học các patterns phức tạp.

#### 1.3.1 Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Miền giá trị:** $(0, 1)$
- **Đạo hàm:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Ưu điểm:** Output có thể diễn giải như xác suất
- **Nhược điểm:** Vanishing gradient khi $|z|$ lớn

#### 1.3.2 Tanh (Hyperbolic Tangent)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

- **Miền giá trị:** $(-1, 1)$
- **Đạo hàm:** $\tanh'(z) = 1 - \tanh^2(z)$
- **Ưu điểm:** Zero-centered, tốt hơn sigmoid cho hidden layers
- **Nhược điểm:** Vẫn có vanishing gradient

#### 1.3.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

- **Miền giá trị:** $[0, +\infty)$
- **Đạo hàm:** $\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$
- **Ưu điểm:** Không bị vanishing gradient (khi $z > 0$), tính toán nhanh
- **Nhược điểm:** Dead neurons (khi $z \leq 0$ luôn)

#### 1.3.4 Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

Với $\alpha$ thường = 0.01. Giải quyết vấn đề dead neurons của ReLU.

### 1.4 Forward Propagation (Lan truyền tiến)

Quá trình tính toán từ input đến output:

**Bước 1:** Khởi tạo $\mathbf{a}^{(0)} = \mathbf{x}$

**Bước 2:** Với mỗi lớp $l = 1, 2, ..., L$:
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)})$$

**Bước 3:** Output: $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$

### 1.5 Loss Functions (Hàm mất mát)

#### 1.5.1 Binary Cross-Entropy (cho bài toán phân loại nhị phân)

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Trong đó:
- $N$: Số mẫu
- $y_i \in \{0, 1\}$: Nhãn thực
- $\hat{y}_i \in (0, 1)$: Xác suất dự đoán

#### 1.5.2 Categorical Cross-Entropy (cho bài toán đa lớp)

$$\mathcal{L}_{CCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Với $C$ là số lớp.

#### 1.5.3 Mean Squared Error (cho bài toán hồi quy)

$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

### 1.6 Backpropagation (Lan truyền ngược)

Backpropagation tính gradient của loss function theo các tham số bằng chain rule.

#### 1.6.1 Chain Rule

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \cdot \frac{\partial \mathbf{a}^{(L)}}{\partial \mathbf{a}^{(L-1)}} \cdot ... \cdot \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$$

#### 1.6.2 Công thức cập nhật

Định nghĩa "error signal" của lớp $l$:
$$\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

**Cho output layer ($l = L$):**
$$\boldsymbol{\delta}^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \odot f'^{(L)}(\mathbf{z}^{(L)})$$

**Cho hidden layers ($l = L-1, ..., 1$):**
$$\boldsymbol{\delta}^{(l)} = \left((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right) \odot f'^{(l)}(\mathbf{z}^{(l)})$$

**Gradient của weights và biases:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

### 1.7 Gradient Descent và Các Biến Thể

#### 1.7.1 Vanilla Gradient Descent

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

Trong đó $\eta$ là learning rate.

#### 1.7.2 Stochastic Gradient Descent (SGD)

Cập nhật trên từng mini-batch thay vì toàn bộ dataset:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; \mathbf{x}^{(i:i+m)}, \mathbf{y}^{(i:i+m)})$$

#### 1.7.3 SGD với Momentum

$$\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \mathbf{v}_t$$

Với $\gamma$ (thường = 0.9) là hệ số momentum.

#### 1.7.4 Adam (Adaptive Moment Estimation)

Kết hợp momentum và adaptive learning rate:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L})^2$$

Bias correction:
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

Update rule:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Hyperparameters mặc định: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### 1.8 Regularization Techniques

#### 1.8.1 L2 Regularization (Weight Decay)

Thêm penalty vào loss function:

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2} \sum_{l} ||\mathbf{W}^{(l)}||_F^2$$

Trong đó $||\cdot||_F$ là Frobenius norm và $\lambda$ là hệ số regularization.

**Tác dụng:** Giữ weights nhỏ, giảm overfitting.

#### 1.8.2 Dropout

Trong training, ngẫu nhiên "tắt" mỗi neuron với xác suất $p$:

$$\tilde{\mathbf{a}}^{(l)} = \mathbf{r}^{(l)} \odot \mathbf{a}^{(l)}$$

Với $\mathbf{r}^{(l)} \sim \text{Bernoulli}(1-p)$

**Trong inference:** Scale output bằng $(1-p)$ hoặc sử dụng inverted dropout.

**Tác dụng:** Tạo ensemble effect, giảm co-adaptation giữa các neurons.

#### 1.8.3 Batch Normalization

Chuẩn hóa activations trong mỗi mini-batch:

**Bước 1:** Tính mean và variance của batch:
$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

**Bước 2:** Normalize:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Bước 3:** Scale và shift (learnable parameters):
$$y_i = \gamma \hat{x}_i + \beta$$

**Tác dụng:**
- Giảm internal covariate shift
- Cho phép sử dụng learning rate cao hơn
- Có tác dụng regularization nhẹ
- Accelerate training

### 1.9 Weight Initialization

#### 1.9.1 Xavier/Glorot Initialization

Cho sigmoid/tanh activation:

$$\mathbf{W} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

hoặc uniform:

$$\mathbf{W} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

#### 1.9.2 He Initialization

Cho ReLU activation:

$$\mathbf{W} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**Lý do:** ReLU "giết" một nửa neurons (phần âm), nên cần variance lớn hơn.

---

## 2. Residual Networks (ResNet)

### 2.1 Vấn đề Vanishing/Exploding Gradient

Khi mạng neural trở nên sâu (nhiều layers), gradient có xu hướng:

- **Vanishing:** Gradient $\rightarrow 0$ khi backpropagate qua nhiều layers
- **Exploding:** Gradient $\rightarrow \infty$

#### 2.1.1 Phân tích toán học

Xét chain rule qua $L$ layers:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \cdot \prod_{l=2}^{L} \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}}$$

Nếu $\left|\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}}\right| < 1$ cho mọi $l$:
$$\prod_{l=2}^{L} \left|\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}}\right| \rightarrow 0 \text{ khi } L \rightarrow \infty$$

**Hậu quả:**
- Layers đầu học rất chậm hoặc không học được
- Mạng sâu có thể perform kém hơn mạng nông

### 2.2 Ý tưởng chính của Residual Learning

#### 2.2.1 Residual Block

Thay vì học mapping trực tiếp $\mathcal{H}(\mathbf{x})$, ResNet học **residual function**:

$$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$$

Do đó:
$$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

```
        ┌─────────────────────────────────────┐
        │                                     │
        │         RESIDUAL BLOCK              │
        │                                     │
   x ───┼───┬─────────────────────────────────┼─── x + F(x)
        │   │                                 │
        │   │    ┌──────────────────┐         │
        │   └───►│   F(x) = W₂σ(W₁x)│────┐    │
        │        └──────────────────┘    │    │
        │              Residual          │    │
        │                                │    │
        │        ┌───────────────────────┼────┤
        │        │    Identity (Skip)    │    │
        └────────┴───────────────────────┴────┘
                          │
                          ▼
                    Output = F(x) + x
```

#### 2.2.2 Công thức Residual Block

**Building block cơ bản:**

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

Với 2 layers:
$$\mathcal{F} = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \cdot \mathbf{x})$$

**Với Batch Normalization:**
$$\mathcal{F}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{BN}(\sigma(\text{BN}(\mathbf{W}_1 \cdot \mathbf{x})))$$

### 2.3 Tại sao Residual Learning hiệu quả?

#### 2.3.1 Dễ học Identity Mapping

Nếu identity mapping là optimal ($\mathcal{H}(\mathbf{x}) = \mathbf{x}$), thì:
- **Mạng thường:** Phải học $\mathcal{H}(\mathbf{x}) = \mathbf{x}$ → Khó với multiple nonlinear layers
- **ResNet:** Chỉ cần học $\mathcal{F}(\mathbf{x}) = 0$ → Dễ hơn (push weights về 0)

#### 2.3.2 Gradient Flow tốt hơn

Xét gradient qua residual block:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$$

**Quan trọng:** Số hạng **"1"** đảm bảo gradient luôn được truyền trực tiếp về layers trước, không phụ thuộc vào $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$.

#### 2.3.3 Phân tích qua nhiều layers

Với $L$ residual blocks:

$$\mathbf{x}_L = \mathbf{x}_0 + \sum_{i=0}^{L-1} \mathcal{F}_i(\mathbf{x}_i)$$

Gradient:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left(1 + \frac{\partial}{\partial \mathbf{x}_0}\sum_{i=0}^{L-1} \mathcal{F}_i\right)$$

Số hạng "1" đảm bảo gradient không vanish ngay cả khi các $\frac{\partial \mathcal{F}_i}{\partial \mathbf{x}}$ rất nhỏ.

### 2.4 Projection Shortcut

Khi dimensions không khớp ($\mathbf{x}$ và $\mathcal{F}(\mathbf{x})$ có size khác nhau):

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{W}_s \mathbf{x}$$

Với $\mathbf{W}_s$ là linear projection (thường là $1 \times 1$ convolution hoặc dense layer).

### 2.5 Pre-activation Residual Block

Cải tiến bởi He et al. (2016) - đặt BN và ReLU trước weight layer:

**Original:**
$$\mathbf{y} = \sigma(\mathcal{F}(\mathbf{x}) + \mathbf{x})$$

**Pre-activation:**
$$\mathbf{y} = \mathcal{F}(\sigma(\text{BN}(\mathbf{x}))) + \mathbf{x}$$

**Ưu điểm:** 
- Identity path hoàn toàn "sạch" (không qua nonlinearity)
- Gradient flow tốt hơn nữa

### 2.6 So sánh DNN thường vs ResNet

| Aspect | Plain DNN | ResNet |
|--------|-----------|--------|
| Gradient Flow | Qua tất cả layers | Có đường tắt (skip) |
| Vanishing Gradient | Nghiêm trọng khi sâu | Được giải quyết |
| Depth | Giới hạn (~20 layers) | Có thể rất sâu (100+ layers) |
| Learning | Học $\mathcal{H}(\mathbf{x})$ trực tiếp | Học residual $\mathcal{F}(\mathbf{x})$ |
| Optimization | Khó hơn | Dễ hơn |

### 2.7 Triển khai Residual Block (Pseudocode)

```python
def residual_block(x, units, dropout_rate=0.3):
    """
    Residual block with 2 dense layers
    
    Args:
        x: Input tensor
        units: Number of neurons
        dropout_rate: Dropout probability
    
    Returns:
        Output tensor with skip connection
    """
    # Store input for skip connection
    shortcut = x
    
    # First dense layer
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second dense layer
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    
    # Projection if dimensions don't match
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)
    
    # Add skip connection
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    
    return x
```

---

## 3. Áp dụng trong Intrusion Detection

### 3.1 Tại sao Deep Learning phù hợp?

1. **Automatic Feature Learning:** DNN tự động học các features phức tạp từ raw network traffic data
2. **Non-linear Patterns:** Attacks thường có patterns phi tuyến phức tạp
3. **Scalability:** Xử lý được lượng data lớn trong môi trường production
4. **Generalization:** Có khả năng phát hiện zero-day attacks (attacks chưa từng thấy)

### 3.2 Kiến trúc được sử dụng

**Deep Neural Network:**
```
Input (N features) 
  → Dense(512) + BN + ReLU + Dropout(0.3)
  → Dense(256) + BN + ReLU + Dropout(0.3)
  → Dense(128) + BN + ReLU + Dropout(0.3)
  → Dense(64) + BN + ReLU + Dropout(0.3)
  → Dense(32) + BN + ReLU + Dropout(0.3)
  → Dense(1) + Sigmoid
  → Output (0: Normal, 1: Attack)
```

**Residual DNN:**
```
Input (N features)
  → Initial Projection to 256
  → Residual Block (256 units) with skip connection
  → Residual Block (128 units) with skip connection  
  → Residual Block (64 units) with skip connection
  → Dense(1) + Sigmoid
  → Output (0: Normal, 1: Attack)
```

### 3.3 Hyperparameters và Lý do

| Hyperparameter | Value | Lý do |
|----------------|-------|-------|
| Hidden layers | (512, 256, 128, 64, 32) | Giảm dần để extract hierarchical features |
| Activation | ReLU | Tránh vanishing gradient, tính toán nhanh |
| Dropout | 0.3 | Giảm overfitting, phù hợp với dataset size |
| L2 Regularization | $10^{-4}$ | Kiểm soát model complexity |
| Batch Normalization | After each Dense | Ổn định training, accelerate convergence |
| Optimizer | Adam | Adaptive learning rate, momentum |
| Learning Rate | $10^{-3}$ | Default cho Adam, có reduction schedule |
| Batch Size | 256 | Trade-off giữa speed và gradient accuracy |

---

## 4. Tài liệu tham khảo

1. **Rosenblatt, F. (1958).** The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386.

2. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

3. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. *CVPR*, 770-778.

4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Identity mappings in deep residual networks. *ECCV*, 630-645.

5. **Ioffe, S., & Szegedy, C. (2015).** Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 448-456.

6. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929-1958.

7. **Kingma, D. P., & Ba, J. (2015).** Adam: A method for stochastic optimization. *ICLR*.

8. **Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 249-256.

---

*Tài liệu này được viết cho mục đích học thuật và nghiên cứu.*

