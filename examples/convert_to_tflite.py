"""
Android TFLite Model Converter
Bu script BERT modelini TensorFlow Lite formatına dönüştürür
"""

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np

def convert_to_tflite(model_name, save_path, quantize=True):
    """
    BERT modelini TFLite formatına dönüştürür
    
    Args:
        model_name: Hugging Face model adı
        save_path: Kaydedilecek .tflite dosya yolu
        quantize: Quantization uygulansın mı (FP16)
    """
    print(f"Loading model: {model_name}")
    model = TFBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # SavedModel formatında kaydet
    print("Saving as SavedModel...")
    model.save_pretrained('./temp_saved_model', saved_model=True)
    
    # TFLite converter oluştur
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model('./temp_saved_model')
    
    # Optimizasyonlar
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Quantization: FP16")
    
    # Dönüştür
    tflite_model = converter.convert()
    
    # Kaydet
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\nModel saved: {save_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    return save_path

def test_tflite_model(tflite_path, model_name):
    """TFLite modelini test eder"""
    print(f"\nTesting TFLite model: {tflite_path}")
    
    # Interpreter yükle
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Input/output detayları
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test input
    tokenizer = BertTokenizer.from_pretrained(model_name)
    test_text = "This movie is great!"
    inputs = tokenizer(
        test_text,
        return_tensors='np',
        padding='max_length',
        max_length=128,
        truncation=True
    )
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], inputs['input_ids'])
    interpreter.set_tensor(input_details[1]['index'], inputs['attention_mask'])
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output, axis=-1)[0]
    
    print(f"Test text: {test_text}")
    print(f"Prediction: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")
    print(f"Logits: {output[0]}")

if __name__ == "__main__":
    # DistilBERT modelini dönüştür
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    output_path = "distilbert_sentiment.tflite"
    
    # Dönüştür
    tflite_path = convert_to_tflite(
        model_name=model_name,
        save_path=output_path,
        quantize=True
    )
    
    # Test et
    test_tflite_model(tflite_path, model_name)
    
    print("\n✅ Conversion completed!")
    print(f"   Model ready for Android deployment: {output_path}")
