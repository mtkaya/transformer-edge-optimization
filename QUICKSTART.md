# 🚀 Hızlı Başlangıç Rehberi

## Kurulum (5 dakika)

### 1. Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/transformer-edge-optimization.git
cd transformer-edge-optimization
```

### 2. Virtual Environment Oluşturun

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 3. Dependencies Yükleyin

```bash
pip install -r requirements.txt
```

## 📓 Notebook'ları Çalıştırma

### Google Colab'de (Önerilen)

1. Her notebook'un üstündeki "Open in Colab" butonuna tıklayın
2. Runtime → Change runtime type → GPU seçin
3. Hücreleri sırayla çalıştırın

### Lokal Jupyter

```bash
jupyter notebook notebooks/
```

## 🎯 İlk Örnek: Quantization

```python
import torch
from transformers import BertForSequenceClassification

# Model yükle
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Model boyutunu karşılaştır
print(f"Orijinal: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"Quantized: ~%75 daha küçük")
```

## 📱 Android Deployment

### 1. Model Dönüştürme

```bash
python examples/convert_to_tflite.py
```

Bu komut `distilbert_sentiment.tflite` dosyası oluşturur.

### 2. Android Projesine Ekleme

```kotlin
// 1. TFLite dosyasını assets/ klasörüne kopyalayın
// 2. build.gradle'a dependency ekleyin:
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
}

// 3. Kullanım:
val analyzer = SentimentAnalyzer(context)
val result = analyzer.predict("This is amazing!")
println(result.label)  // POSITIVE veya NEGATIVE
```

## 🌐 Web Deployment

```javascript
// Transformers.js ile tarayıcıda çalıştırma
import { pipeline } from '@xenova/transformers';

const classifier = await pipeline('sentiment-analysis');
const result = await classifier('I love this!');
console.log(result);  // [{ label: 'POSITIVE', score: 0.9998 }]
```

## 📊 Notebook Listesi

| # | Notebook | Süre | Seviye |
|---|----------|------|--------|
| 01 | [Quantization Basics](notebooks/01_quantization_basics.ipynb) | 15 dk | Başlangıç |
| 02 | [Hugging Face Optimum](notebooks/02_huggingface_optimum.ipynb) | 20 dk | Orta |
| 05 | [Knowledge Distillation](notebooks/05_distilbert_training.ipynb) | 30 dk | İleri |

## 🔥 Popüler Örnekler

### Model Boyutunu Küçültme (4x)

```python
# Quantization
quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

### Hızı Artırma (2x)

```python
# ONNX Runtime
from optimum.onnxruntime import ORTModelForSequenceClassification
model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
```

### Küçük Model Eğitme (3x küçük)

```python
# Knowledge Distillation
loss = distillation_loss(student_logits, teacher_logits, labels)
```

## 🐛 Sorun Giderme

### CUDA out of memory

```python
# Batch size'ı küçültün
batch_size = 8  # 32 yerine
```

### Import hatası

```bash
# Tüm dependencies'i tekrar yükleyin
pip install -r requirements.txt --force-reinstall
```

### Notebook çalışmıyor

```bash
# Jupyter'i güncelleyin
pip install --upgrade jupyter notebook
```

## 📚 Öğrenme Yolu

### Başlangıç (1-2 gün)
1. ✅ Quantization Basics notebook
2. ✅ Model boyutu karşılaştırması
3. ✅ Basit Android örneği

### Orta (1 hafta)
1. ✅ Hugging Face Optimum
2. ✅ ONNX dönüşümleri
3. ✅ TFLite deployment

### İleri (2-3 hafta)
1. ✅ Knowledge Distillation
2. ✅ Pruning teknikleri
3. ✅ Custom optimization pipeline

## 💡 Pro Tips

1. **GPU kullanın**: Colab'de ücretsiz GPU
2. **Küçük başlayın**: DistilBERT ile başlayın
3. **Benchmark edin**: Her optimizasyondan sonra ölçün
4. **Combine teknikleri**: Distillation + Quantization
5. **Real device test**: Emulator yerine gerçek cihaz

## 🆘 Yardım

- 📖 [Full Documentation](README.md)
- 💬 [GitHub Discussions](https://github.com/yourusername/transformer-edge-optimization/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/transformer-edge-optimization/issues)

## ⭐ Sonraki Adımlar

- [ ] 3 notebook'u tamamlayın
- [ ] İlk modelinizi quantize edin
- [ ] Android'de test edin
- [ ] Kendi modelinizi optimize edin
- [ ] Topluluğa katkıda bulunun

---

**Başarılar!** 🎉 Sorularınız için issue açmaktan çekinmeyin.
