package com.ricedetector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * TFLite Classifier for Rice Quality Detection
 */
class TFLiteClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    companion object {
        private const val TAG = "TFLiteClassifier"

        // Model configuration - ADJUST TO MATCH YOUR MODEL
        private const val MODEL_FILE = "rice_classifier.tflite"
        private const val LABELS_FILE = "classes.txt"

        // Input dimensions - MUST MATCH YOUR MODEL
        const val INPUT_WIDTH = 128
        const val INPUT_HEIGHT = 128
        const val INPUT_CHANNELS = 3

        // Normalization parameters
        const val IMAGE_MEAN = 0f
        const val IMAGE_STD = 255f
    }

    data class Classification(
        val label: String,
        val confidence: Float,
        val inferenceTime: Long,
        val allProbabilities: FloatArray? = null
    )

    /**
     * Initialize the classifier
     */
    fun initialize() {
        try {
            Log.d(TAG, "Loading TFLite model...")

            // Load model
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_FILE)

            // Configure interpreter options
            val options = Interpreter.Options().apply {
                setNumThreads(4)  // Use 4 threads
                setUseNNAPI(true) // Use Android Neural Networks API if available
            }

            interpreter = Interpreter(modelBuffer, options)

            // Log model details
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()

            Log.d(TAG, "Model loaded successfully")
            Log.d(TAG, "Input shape: ${inputShape?.contentToString()}")
            Log.d(TAG, "Output shape: ${outputShape?.contentToString()}")

            // Load labels
            loadLabels()

        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            throw RuntimeException("Failed to load TFLite model: ${e.message}")
        }
    }

    /**
     * Load class labels from assets
     */
    private fun loadLabels() {
        try {
            val reader = BufferedReader(InputStreamReader(context.assets.open(LABELS_FILE)))
            labels = reader.readLines().filter { it.isNotBlank() }
            reader.close()

            Log.d(TAG, "Loaded ${labels.size} labels: $labels")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels", e)
            // Fallback labels
            labels = List(3) { "Class $it" }
        }
    }

    /**
     * Preprocess image for model input
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize bitmap to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_WIDTH,
            INPUT_HEIGHT,
            true
        )

        // Check if model expects quantized input
        val inputDataType = interpreter?.getInputTensor(0)?.dataType()
        val isQuantized = inputDataType == org.tensorflow.lite.DataType.UINT8

        // Allocate buffer
        val inputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS)
        } else {
            ByteBuffer.allocateDirect(4 * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS)
        }.apply {
            order(ByteOrder.nativeOrder())
        }

        // Extract pixel values
        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        resizedBitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        // Convert pixels to input format
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            if (isQuantized) {
                // UINT8 quantized (0-255)
                inputBuffer.put(r.toByte())
                inputBuffer.put(g.toByte())
                inputBuffer.put(b.toByte())
            } else {
                // Float32 normalized
                inputBuffer.putFloat((r - IMAGE_MEAN) / IMAGE_STD)
                inputBuffer.putFloat((g - IMAGE_MEAN) / IMAGE_STD)
                inputBuffer.putFloat((b - IMAGE_MEAN) / IMAGE_STD)
            }
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    /**
     * Run inference on bitmap
     */
    fun classify(bitmap: Bitmap): Classification {
        if (interpreter == null) {
            throw IllegalStateException("Model not initialized. Call initialize() first.")
        }

        val startTime = System.currentTimeMillis()

        // Preprocess image
        val inputBuffer = preprocessImage(bitmap)

        // Prepare output buffer
        val outputDataType = interpreter?.getOutputTensor(0)?.dataType()
        val isQuantized = outputDataType == org.tensorflow.lite.DataType.UINT8

        val outputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(labels.size)
        } else {
            ByteBuffer.allocateDirect(4 * labels.size)
        }.apply {
            order(ByteOrder.nativeOrder())
        }

        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // Parse output
        val probabilities = if (isQuantized) {
            // Dequantize UINT8 output
            val scale = interpreter?.getOutputTensor(0)?.quantizationParams()?.scale ?: 1f
            val zeroPoint = interpreter?.getOutputTensor(0)?.quantizationParams()?.zeroPoint ?: 0

            FloatArray(labels.size) { i ->
                val quantizedValue = outputBuffer.get().toInt() and 0xFF
                (quantizedValue - zeroPoint) * scale
            }
        } else {
            // Read Float32 output
            FloatArray(labels.size) { outputBuffer.float }
        }

        // Apply softmax
        val softmaxProbs = softmax(probabilities)

        // Find top prediction
        val maxIndex = softmaxProbs.indices.maxByOrNull { softmaxProbs[it] } ?: 0
        val maxConfidence = softmaxProbs[maxIndex]

        val inferenceTime = System.currentTimeMillis() - startTime

        Log.d(TAG, "Inference completed in ${inferenceTime}ms")
        Log.d(TAG, "Prediction: ${labels.getOrElse(maxIndex) { "Unknown" }} (${(maxConfidence * 100).toInt()}%)")
        Log.d(TAG, "All probabilities: ${softmaxProbs.contentToString()}")

        return Classification(
            label = labels.getOrElse(maxIndex) { "Unknown" },
            confidence = maxConfidence,
            inferenceTime = inferenceTime,
            allProbabilities = softmaxProbs
        )
    }

    /**
     * Get top N predictions
     */
    fun classifyTopN(bitmap: Bitmap, topN: Int = 3): List<Classification> {
        if (interpreter == null) {
            throw IllegalStateException("Model not initialized")
        }

        val startTime = System.currentTimeMillis()
        val inputBuffer = preprocessImage(bitmap)

        val outputDataType = interpreter?.getOutputTensor(0)?.dataType()
        val isQuantized = outputDataType == org.tensorflow.lite.DataType.UINT8

        val outputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(labels.size)
        } else {
            ByteBuffer.allocateDirect(4 * labels.size)
        }.apply {
            order(ByteOrder.nativeOrder())
        }

        interpreter?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val probabilities = if (isQuantized) {
            val scale = interpreter?.getOutputTensor(0)?.quantizationParams()?.scale ?: 1f
            val zeroPoint = interpreter?.getOutputTensor(0)?.quantizationParams()?.zeroPoint ?: 0
            FloatArray(labels.size) { i ->
                val quantizedValue = outputBuffer.get().toInt() and 0xFF
                (quantizedValue - zeroPoint) * scale
            }
        } else {
            FloatArray(labels.size) { outputBuffer.float }
        }

        val softmaxProbs = softmax(probabilities)
        val inferenceTime = System.currentTimeMillis() - startTime

        return softmaxProbs.indices
            .sortedByDescending { softmaxProbs[it] }
            .take(topN)
            .map { index ->
                Classification(
                    label = labels.getOrElse(index) { "Unknown" },
                    confidence = softmaxProbs[index],
                    inferenceTime = inferenceTime
                )
            }
    }

    /**
     * Softmax function
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }

    /**
     * Benchmark inference time
     */
    fun benchmark(bitmap: Bitmap, iterations: Int = 10): Float {
        val times = mutableListOf<Long>()

        repeat(iterations) {
            val start = System.currentTimeMillis()
            classify(bitmap)
            times.add(System.currentTimeMillis() - start)
        }

        val avgTime = times.average().toFloat()
        Log.d(TAG, "Average inference time over $iterations runs: ${avgTime}ms")

        return avgTime
    }

    /**
     * Clean up resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        Log.d(TAG, "Classifier closed")
    }
}