package com.ricedetector

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Manages all three TFLite models
 */
class ModelManager(private val context: Context) {

    // Model types
    enum class ModelType {
        GRAIN_CLASSIFICATION,
        DEFECT_CLASSIFICATION
    }

    // Model configurations
    data class ModelConfig(
        val modelFile: String,
        val labelsFile: String,
        val inputSize: Int,
        val isDetection: Boolean = false
    )

    private val modelConfigs = mapOf(
        ModelType.GRAIN_CLASSIFICATION to ModelConfig(
            modelFile = "rice_classifier.tflite",
            labelsFile = "grain_labels.txt",
            inputSize = 128
        ),
        ModelType.DEFECT_CLASSIFICATION to ModelConfig(
            modelFile = "rice_defect_classifier_int8.tflite",
            labelsFile = "defect_labels.txt",
            inputSize = 150
        )
    )

    // Store interpreters and labels
    private val interpreters = mutableMapOf<ModelType, Interpreter>()
    private val labels = mutableMapOf<ModelType, List<String>>()

    companion object {
        private const val TAG = "ModelManager"
    }

    /**
     * Initialize all models
     */
    suspend fun initializeAllModels() {
        Log.d(TAG, "Initializing all models...")

        modelConfigs.forEach { (type, config) ->
            try {
                // Load model
                val modelBuffer = FileUtil.loadMappedFile(context, config.modelFile)
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
//                    setUseNNAPI(true)
                }
                val interpreter = Interpreter(modelBuffer, options)
                interpreters[type] = interpreter

                // Load labels
                val labelsList = loadLabels(config.labelsFile)
                labels[type] = labelsList

                Log.d(TAG, "✅ Loaded ${type.name}: ${config.modelFile} (${labelsList.size} classes)")

            } catch (e: Exception) {
                Log.e(TAG, "❌ Failed to load ${type.name}: ${e.message}", e)
                throw e
            }
        }

        Log.d(TAG, "All models initialized successfully!")
    }

    /**
     * Load labels from file
     */
    private fun loadLabels(filename: String): List<String> {
        val reader = BufferedReader(InputStreamReader(context.assets.open(filename)))
        return reader.readLines().filter { it.isNotBlank() }
    }

    /**
     * Get interpreter for model type
     */
    fun getInterpreter(type: ModelType): Interpreter {
        return interpreters[type] ?: throw IllegalStateException("Model $type not initialized")
    }

    /**
     * Get labels for model type
     */
    fun getLabels(type: ModelType): List<String> {
        return labels[type] ?: throw IllegalStateException("Labels for $type not loaded")
    }

    /**
     * Get config for model type
     */
    fun getConfig(type: ModelType): ModelConfig {
        return modelConfigs[type] ?: throw IllegalStateException("Config for $type not found")
    }

    /**
     * Clean up resources
     */
    fun close() {
        interpreters.values.forEach { it.close() }
        interpreters.clear()
        labels.clear()
        Log.d(TAG, "All models closed")
    }
}