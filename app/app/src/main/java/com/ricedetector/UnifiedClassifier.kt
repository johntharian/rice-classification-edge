package com.ricedetector

import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * Unified classifier for all three models
 */
class UnifiedClassifier(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val inputSize: Int,
    private val isDetection: Boolean = false
) {

    companion object {
        private const val TAG = "UnifiedClassifier"
        private const val IMAGE_MEAN = 0f
        private const val IMAGE_STD = 255f
    }

    data class ClassificationResult(
        val label: String,
        val confidence: Float,
        val inferenceTime: Long,
        val allProbabilities: FloatArray? = null
    )

    data class DetectionResult(
        val boxes: List<BoundingBox>,
        val inferenceTime: Long
    )

    data class BoundingBox(
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val label: String,
        val confidence: Float
    )

    /**
     * Run classification
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        val startTime = System.currentTimeMillis()

        // Preprocess
        val inputBuffer = preprocessImage(bitmap)

        // Get output data type
        val outputDataType = interpreter.getOutputTensor(0).dataType()
        val isQuantized = outputDataType == DataType.UINT8

        // Prepare output
        val outputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(labels.size)
        } else {
            ByteBuffer.allocateDirect(4 * labels.size)
        }.apply {
            order(ByteOrder.nativeOrder())
        }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // Parse output
        val probabilities = if (isQuantized) {
            val scale = interpreter.getOutputTensor(0).quantizationParams().scale
            val zeroPoint = interpreter.getOutputTensor(0).quantizationParams().zeroPoint
            FloatArray(labels.size) { i ->
                val quantizedValue = outputBuffer.get().toInt() and 0xFF
                (quantizedValue - zeroPoint) * scale
            }
        } else {
            FloatArray(labels.size) { outputBuffer.float }
        }

        // Apply softmax
        val softmaxProbs = softmax(probabilities)

        // Get top prediction
        val maxIndex = softmaxProbs.indices.maxByOrNull { softmaxProbs[it] } ?: 0
        val maxConfidence = softmaxProbs[maxIndex]

        val inferenceTime = System.currentTimeMillis() - startTime

        Log.d(TAG, "Classification: ${labels[maxIndex]} (${(maxConfidence * 100).toInt()}%) in ${inferenceTime}ms")

        return ClassificationResult(
            label = labels[maxIndex],
            confidence = maxConfidence,
            inferenceTime = inferenceTime,
            allProbabilities = softmaxProbs
        )
    }

    /**
     * Preprocess image
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val inputDataType = interpreter.getInputTensor(0).dataType()
        val isQuantized = inputDataType == DataType.UINT8

        val inputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(inputSize * inputSize * 3)
        } else {
            ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        }.apply {
            order(ByteOrder.nativeOrder())
        }

        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            if (isQuantized) {
                inputBuffer.put(r.toByte())
                inputBuffer.put(g.toByte())
                inputBuffer.put(b.toByte())
            } else {
                inputBuffer.putFloat((r - IMAGE_MEAN) / IMAGE_STD)
                inputBuffer.putFloat((g - IMAGE_MEAN) / IMAGE_STD)
                inputBuffer.putFloat((b - IMAGE_MEAN) / IMAGE_STD)
            }
        }

        inputBuffer.rewind()
        return inputBuffer
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
}