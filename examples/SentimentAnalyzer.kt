/**
 * Android TFLite Sentiment Analyzer
 * 
 * DistilBERT modelini kullanarak sentiment analysis yapan Android sınıfı
 * 
 * Kullanım:
 * val analyzer = SentimentAnalyzer(context)
 * val result = analyzer.predict("This is amazing!")
 */

package com.example.transformeredge

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class SentimentResult(
    val label: String,
    val score: Float
)

class SentimentAnalyzer(private val context: Context) {
    
    private lateinit var interpreter: Interpreter
    private val maxSequenceLength = 128
    private val vocabSize = 30522
    
    // DistilBERT vocabulary - basitleştirilmiş tokenizer
    private val tokenizer = SimpleTokenizer()
    
    init {
        loadModel()
    }
    
    /**
     * TFLite modelini yükler
     */
    private fun loadModel() {
        try {
            val modelFile = loadModelFile("distilbert_sentiment.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4) // CPU thread sayısı
                setUseNNAPI(true) // NNAPI acceleration
            }
            interpreter = Interpreter(modelFile, options)
            println("Model loaded successfully")
        } catch (e: Exception) {
            println("Error loading model: ${e.message}")
            throw e
        }
    }
    
    /**
     * Assets'ten model dosyasını yükler
     */
    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Text'i sentiment analysis için tahmin eder
     * 
     * @param text Analiz edilecek metin
     * @return SentimentResult (label ve score)
     */
    fun predict(text: String): SentimentResult {
        // Tokenize
        val tokens = tokenizer.tokenize(text, maxLength = maxSequenceLength)
        
        // Input tensors hazırla
        val inputIds = tokens.inputIds
        val attentionMask = tokens.attentionMask
        
        // ByteBuffer'ları oluştur
        val inputIdsBuffer = createIntBuffer(inputIds)
        val attentionMaskBuffer = createIntBuffer(attentionMask)
        
        // Output buffer
        val outputBuffer = ByteBuffer.allocateDirect(2 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        
        // Inference
        val inputs = arrayOf<Any>(inputIdsBuffer, attentionMaskBuffer)
        val outputs = mapOf(0 to outputBuffer)
        
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        
        // Sonuçları parse et
        outputBuffer.rewind()
        val logits = FloatArray(2) { outputBuffer.float }
        
        // Softmax
        val scores = softmax(logits)
        val predictedClass = if (scores[1] > scores[0]) 1 else 0
        
        return SentimentResult(
            label = if (predictedClass == 1) "POSITIVE" else "NEGATIVE",
            score = scores[predictedClass]
        )
    }
    
    /**
     * Int array'den ByteBuffer oluşturur
     */
    private fun createIntBuffer(array: IntArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(array.size * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        array.forEach { buffer.putInt(it) }
        buffer.rewind()
        return buffer
    }
    
    /**
     * Softmax fonksiyonu
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }
    
    /**
     * Cleanup
     */
    fun close() {
        interpreter.close()
    }
}

/**
 * Basitleştirilmiş tokenizer
 * Gerçek uygulamada Hugging Face tokenizer kullanılmalı
 */
data class TokenizerOutput(
    val inputIds: IntArray,
    val attentionMask: IntArray
)

class SimpleTokenizer {
    private val clsTokenId = 101
    private val sepTokenId = 102
    private val padTokenId = 0
    
    fun tokenize(text: String, maxLength: Int): TokenizerOutput {
        // Basit whitespace tokenization (gerçekte WordPiece kullanılmalı)
        val words = text.lowercase().split(" ")
        val tokens = mutableListOf(clsTokenId)
        
        // Her kelimeyi token ID'ye çevir (basitleştirilmiş)
        words.forEach { word ->
            val tokenId = word.hashCode().absoluteValue % 30522
            tokens.add(tokenId)
        }
        
        tokens.add(sepTokenId)
        
        // Padding
        val inputIds = IntArray(maxLength) { padTokenId }
        val attentionMask = IntArray(maxLength) { 0 }
        
        tokens.take(maxLength).forEachIndexed { index, tokenId ->
            inputIds[index] = tokenId
            attentionMask[index] = 1
        }
        
        return TokenizerOutput(inputIds, attentionMask)
    }
}

// Kullanım örneği
/*
class MainActivity : AppCompatActivity() {
    private lateinit var sentimentAnalyzer: SentimentAnalyzer
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Analyzer'ı başlat
        sentimentAnalyzer = SentimentAnalyzer(this)
        
        // Test
        val text = "This movie is absolutely fantastic!"
        val result = sentimentAnalyzer.predict(text)
        
        println("Text: $text")
        println("Sentiment: ${result.label}")
        println("Confidence: ${result.score}")
    }
    
    override fun onDestroy() {
        super.onDestroy()
        sentimentAnalyzer.close()
    }
}
*/
