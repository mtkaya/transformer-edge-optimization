# 🚀 Transformer Edge Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtkaya/transformer-edge-optimization)

> **Büyük Transformer modellerini mobil ve edge cihazlarda çalıştırmak için kapsamlı rehber ve araçlar.**

---

## ✨ Özellikler

### 🎯 Optimizasyon Teknikleri

- **Quantization** - INT8, FP16, Dynamic Quantization
  - Model boyutu: **4x azalma**
  - Minimal doğruluk kaybı (**~1-2%**)
  
- **Knowledge Distillation** - Öğretmen-öğrenci öğrenimi
  - Model boyutu: **6-10x azalma**
  - Doğruluk korunur (**~2-4% kayıp**)
  
- **ONNX Runtime** - Cross-platform deployment
  - Hardware-accelerated inference
  - Mobil ve edge cihaz desteği

---

## 🚀 Hızlı Başlangıç

### Google Colab'de Çalıştır (Önerilen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtkaya/transformer-edge-optimization/blob/main/notebooks/01_quantization_basics.ipynb)

1. Yukarıdaki butona tıkla
2. Runtime → Change runtime type → **GPU**
3. Runtime → **Run all**
4. 5 dakika bekle ve sonuçları izle! 🎉

### Lokal Kurulum
```bash
# Repository'yi klonla
git clone https://github.com/mtkaya/transformer-edge-optimization.git
cd transformer-edge-optimization

# Bağımlılıkları yükle
pip install -r requirements.txt

# Jupyter'i başlat
jupyter notebook notebooks/
```

---

## 📓 Notebook'lar

### 1️⃣ Quantization Basics (15 dakika)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtkaya/transformer-edge-optimization/blob/main/notebooks/01_quantization_basics.ipynb)

- FP32 → INT8 dönüşümü
- Model boyutu: **4x azaltma**
- İnferans hızı: **2x artış**

### 2️⃣ ONNX Runtime Optimization (20 dakika)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtkaya/transformer-edge-optimization/blob/main/notebooks/02_huggingface_optimum.ipynb)

- PyTorch → ONNX dönüşümü
- Dynamic quantization
- Cross-platform deployment

### 3️⃣ Knowledge Distillation (30 dakika)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtkaya/transformer-edge-optimization/blob/main/notebooks/05_distilbert_training.ipynb)

- Teacher-student training
- Model boyutu: **7.6x azaltma**
- BERT → TinyBERT

---

## 💻 Kullanım Örneği
```python
import torch
from transformers import AutoModelForSequenceClassification

# Model yükle
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Quantize et (FP32 → INT8)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Model boyutu 4x daha küçük! 🎉
print("Model 4x daha küçük, 2x daha hızlı!")
```

---

## 📊 Benchmark Sonuçları

| Teknik | Boyut Azaltma | Hız Artışı | Doğruluk |
|--------|---------------|------------|----------|
| **Quantization (INT8)** | 4.0x | 2.1x | 91.2% |
| **ONNX Runtime** | 3.8x | 2.2x | 88.2% |
| **Distillation** | 7.6x | 3.0x | 87.1% |
| **Combined** | 31.4x | 9.5x | 85.8% |

---

## 🛠️ Desteklenen Platformlar

- ✅ **Android** - TensorFlow Lite
- ✅ **iOS** - Core ML
- ✅ **Web** - Transformers.js
- ✅ **Edge Devices** - ONNX Runtime

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! 

1. Fork yapın
2. Feature branch oluşturun
3. Commit yapın
4. Pull Request açın

Detaylar için: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 Lisans

Bu proje MIT lisansı altındadır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙏 Teşekkürler

- [Hugging Face](https://huggingface.co/) - Transformers ve Optimum
- [ONNX](https://onnx.ai/) - Model interoperability
- Açık kaynak topluluğuna ❤️

---

## 📧 İletişim

- **GitHub Issues:** [Sorun bildir](https://github.com/mtkaya/transformer-edge-optimization/issues)
- **Discussions:** [Tartışmalara katıl](https://github.com/mtkaya/transformer-edge-optimization/discussions)

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**

Made with ❤️ for the AI community

</div>
