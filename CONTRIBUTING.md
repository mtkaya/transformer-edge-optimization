# Katkıda Bulunma Rehberi

Transformer Edge Optimization projesine katkıda bulunmak istediğiniz için teşekkürler! 🎉

## 🚀 Nasıl Katkıda Bulunulur

### 1. Repository'yi Fork Edin

- GitHub'da bu repository'nin sağ üst köşesindeki "Fork" butonuna tıklayın
- Fork'u kendi hesabınıza klonlayın

### 2. Development Branch Oluşturun

```bash
git checkout -b feature/amazing-feature
```

Branch isimlendirme konvansiyonları:
- `feature/` - Yeni özellikler için
- `bugfix/` - Bug düzeltmeleri için
- `docs/` - Dokümantasyon güncellemeleri için
- `refactor/` - Code refactoring için

### 3. Değişikliklerinizi Yapın

- Kod stiline uyun (PEP 8 for Python)
- Açıklayıcı commit mesajları yazın
- Mümkünse test ekleyin

### 4. Test Edin

```bash
# Python testleri
pytest tests/

# Notebook'ları test edin
jupyter nbconvert --execute notebooks/*.ipynb
```

### 5. Commit ve Push

```bash
git add .
git commit -m "feat: Add amazing feature"
git push origin feature/amazing-feature
```

Commit mesaj formatı:
- `feat:` - Yeni özellik
- `fix:` - Bug fix
- `docs:` - Dokümantasyon
- `style:` - Formatlama, noktalama
- `refactor:` - Code refactoring
- `test:` - Test ekleme/düzenleme
- `chore:` - Maintenance

### 6. Pull Request Oluşturun

- GitHub'da "Pull Request" açın
- Değişikliklerinizi detaylı açıklayın
- İlgili issue'ları reference edin

## 📋 Katkı Alanları

### Notebook'lar
- Yeni optimizasyon teknikleri
- Benchmark'lar
- Görselleştirmeler
- Tutorial'lar

### Kod Örnekleri
- Android örnekleri
- iOS örnekleri
- Web deployment
- Edge device implementasyonları

### Dokümantasyon
- README güncellemeleri
- API dokümantasyonu
- Tutorial'lar
- Çeviriler

### Araçlar ve Utilities
- Model dönüştürme script'leri
- Benchmark araçları
- CI/CD pipeline'ları

## ✅ Code Review Süreci

1. Otomatik testler çalışır
2. Maintainer'lar kod review yapar
3. Gerekirse değişiklik talep edilir
4. Onaylandıktan sonra merge edilir

## 🎨 Code Style

### Python
```python
# PEP 8 uyumlu
# Type hints kullanın
def quantize_model(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """
    Model quantization uygular.
    
    Args:
        model: PyTorch model
        dtype: Target data type
        
    Returns:
        Quantized model
    """
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)
```

### Jupyter Notebooks
- Her hücrede açıklayıcı markdown
- Output'lar temizlenmeli (büyük olanlar)
- Reproducible results için seed set edin

## 🐛 Bug Reports

Bug bulduğunuzda lütfen şunları ekleyin:
- Kısa açıklayıcı başlık
- Detaylı açıklama
- Repro adımları
- Beklenen davranış
- Mevcut davranış
- Environment bilgileri (Python version, OS, etc.)
- Screenshots/logs (varsa)

## 💡 Feature Requests

Yeni özellik önerirken:
- Özelliğin ne yaptığını açıklayın
- Neden gerekli olduğunu anlatın
- Varsa örnek kullanım senaryoları
- Alternatif çözümler

## 📫 İletişim

- Issues: GitHub Issues kullanın
- Discussions: GitHub Discussions
- Email: [email eklenecek]

## 🙏 Teşekkürler

Zamanınızı ayırıp katkıda bulunduğunuz için teşekkürler!

---

**Not**: Bu proje [Code of Conduct](CODE_OF_CONDUCT.md)'a tabidir. Katkıda bulunarak bu kurallara uymayı kabul edersiniz.
