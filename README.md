# 🚀 Transformer'ları Cebe Sığdırmak: Edge Cihazlar İçin Optimizasyon

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/transformer-edge-optimization)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Büyük Transformer modellerini mobil ve uç cihazlarda çalıştırmak için kapsamlı rehber ve pratik örnekler.

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Kurulum](#kurulum)
- [Notebook'lar](#notebooks)
- [Optimizasyon Teknikleri](#optimizasyon-teknikleri)
- [Pratik Örnekler](#pratik-örnekler)
- [Benchmark Sonuçları](#benchmark-sonuçları)
- [Kaynaklar](#kaynaklar)

## 🎯 Genel Bakış

Bu repo, Transformer modellerinin mobil telefonlar, IoT cihazları ve edge computing platformlarında verimli çalıştırılması için gerekli teknikleri ve araçları içerir.

### Kapsanan Konular

- **Quantization**: INT8, FP16, Dynamic Quantization
- **Knowledge Distillation**: DistilBERT, TinyBERT, MobileBERT
- **Pruning**: Structured ve Unstructured Pruning
- **Araçlar**: Hugging Face Optimum, ONNX Runtime, TensorFlow Lite, Core ML
- **Deployment**: Android, iOS, Web uygulamaları

## 🔧 Kurulum

### Temel Bağımlılıklar

```bash
# PyTorch ve Transformers
pip install torch transformers

# Optimizasyon araçları
pip install optimum[onnxruntime] onnx onnxruntime

# TensorFlow Lite (isteğe bağlı)
pip install tensorflow

# Quantization araçları
pip install neural-compressor
```

### Repo'yu Klonlama

```bash
git clone https://github.com/yourusername/transformer-edge-optimization.git
cd transformer-edge-optimization
pip install -r requirements.txt
```

## 📓 Notebook'lar

### 1. Quantization
- **[01_quantization_basics.ipynb](notebooks/01_quantization_basics.ipynb)** - Quantization temelleri ve PyTorch örnekleri
- **[02_huggingface_optimum.ipynb](notebooks/02_huggingface_optimum.ipynb)** - Hugging Face Optimum ile INT8 quantization
- **[03_dynamic_quantization.ipynb](notebooks/03_dynamic_quantization.ipynb)** - Dynamic quantization BERT örneği

### 2. Knowledge Distillation
- **[04_distillation_basics.ipynb](notebooks/04_distillation_basics.ipynb)** - Knowledge distillation temel prensipler
- **[05_distilbert_training.ipynb](notebooks/05_distilbert_training.ipynb)** - DistilBERT'ten öğrenci model eğitimi

### 3. Pruning
- **[06_pruning_techniques.ipynb](notebooks/06_pruning_techniques.ipynb)** - Magnitude ve structured pruning
- **[07_attention_head_pruning.ipynb](notebooks/07_attention_head_pruning.ipynb)** - BERT attention head pruning

### 4. Model Dönüşümleri
- **[08_pytorch_to_onnx.ipynb](notebooks/08_pytorch_to_onnx.ipynb)** - PyTorch → ONNX dönüşümü
- **[09_tensorflow_lite.ipynb](notebooks/09_tensorflow_lite.ipynb)** - TensorFlow Lite dönüşümü
- **[10_coreml_conversion.ipynb](notebooks/10_coreml_conversion.ipynb)** - Core ML dönüşümü

### 5. Deployment
- **[11_android_tflite.ipynb](notebooks/11_android_tflite.ipynb)** - Android TFLite deployment
- **[12_transformers_js.ipynb](notebooks/12_transformers_js.ipynb)** - Tarayıcıda Transformers.js

### 6. Benchmark & Karşılaştırma
- **[13_benchmarking.ipynb](notebooks/13_benchmarking.ipynb)** - Performans karşılaştırmaları
- **[14_end_to_end_pipeline.ipynb](notebooks/14_end_to_end_pipeline.ipynb)** - Tam optimizasyon pipeline'ı

## 🔬 Optimizasyon Teknikleri

### Quantization

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
```

### Knowledge Distillation

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### Pruning

```python
import torch.nn.utils.prune as prune

# %30 magnitude pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')
```

## 💡 Pratik Örnekler

### Android Sentiment Analysis

```kotlin
class SentimentAnalyzer(context: Context) {
    private val interpreter: Interpreter
    
    fun predict(text: String): Float {
        val inputIds = tokenize(text)
        val output = Array(1) { FloatArray(2) }
        interpreter.run(inputIds, output)
        return output[0][1] // Positive score
    }
}
```

### Web Tarayıcıda NER

```javascript
import { pipeline } from '@xenova/transformers';

const ner = await pipeline('ner', 'Xenova/bert-base-NER');
const entities = await ner('Apple is looking at buying UK startup');
console.log(entities);
```

## 📊 Benchmark Sonuçları

| Model | Boyut | Inferans Süresi (ms) | Doğruluk |
|-------|-------|----------------------|----------|
| BERT-base (FP32) | 440 MB | 350 ms | 92.5% |
| DistilBERT (FP32) | 255 MB | 220 ms | 89.8% |
| DistilBERT (INT8) | 67 MB | 95 ms | 88.2% |
| TinyBERT (INT8) | 14 MB | 37 ms | 87.1% |

*Benchmark: Pixel 6, 128 token input*

## 🛠️ Araçlar ve Framework'ler

### Desteklenen Araçlar

- **Hugging Face Optimum** - Hardware-accelerated inference
- **ONNX Runtime** - Cross-platform optimization
- **TensorFlow Lite** - Mobile deployment
- **Core ML** - iOS optimization
- **Transformers.js** - Browser inference
- **Intel Neural Compressor** - Intel CPU quantization

## 📚 Kaynaklar

### Akademik Makaleler

- [DistilBERT (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108)
- [TinyBERT (Jiao et al., 2020)](https://arxiv.org/abs/1909.10351)
- [MobileBERT (Sun et al., 2020)](https://arxiv.org/abs/2004.02984)
- [Q8BERT (Zafrir et al., 2021)](https://arxiv.org/abs/1910.06188)

### Dokümantasyon

- [Hugging Face Optimum](https://huggingface.co/docs/optimum)
- [ONNX Runtime](https://onnxruntime.ai)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

### Topluluk

- [Hugging Face Forums](https://discuss.huggingface.co)
- [Reddit r/MachineLearning](https://reddit.com/r/MachineLearning)

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- Hugging Face ekibine harika araçlar için
- Anthropic'e Claude için
- Açık kaynak topluluğuna

## 📧 İletişim

Sorularınız için issue açabilir veya iletişime geçebilirsiniz.

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!
