package com.ricedetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AnalysisActivity : AppCompatActivity() {

    private lateinit var btnBack: Button
    private lateinit var tvModelName: TextView
    private lateinit var ivPreview: ImageView
    private lateinit var progressBar: ProgressBar
    private lateinit var resultsContainer: LinearLayout
    private lateinit var tvResult: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvInferenceTime: TextView
    private lateinit var tvAllProbs: TextView
    private lateinit var btnCamera: Button
    private lateinit var btnGallery: Button

    private lateinit var modelType: ModelManager.ModelType
    private lateinit var classifier: UnifiedClassifier
    private lateinit var modelConfig: ModelManager.ModelConfig

    // Gallery launcher
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handleImageSelection(it) }
    }

    // Camera launcher
    private val takePhoto = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        bitmap?.let { handleCameraBitmap(it) }
    }

    // Permission launcher
    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openGallery()
        } else {
            Toast.makeText(this, "Storage permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    // Camera permission launcher
    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        // Get model type from intent
        val modelTypeName = intent.getStringExtra(MainActivity.EXTRA_MODEL_TYPE)
            ?: run {
                Toast.makeText(this, "Invalid model type", Toast.LENGTH_SHORT).show()
                finish()
                return
            }

        modelType = ModelManager.ModelType.valueOf(modelTypeName)

        // Initialize views
        initViews()

        // Setup classifier
        setupClassifier()

        // Setup listeners
        setupListeners()
    }

    private fun initViews() {
        btnBack = findViewById(R.id.btnBack)
        tvModelName = findViewById(R.id.tvModelName)
        ivPreview = findViewById(R.id.ivPreview)
        progressBar = findViewById(R.id.progressBar)
        resultsContainer = findViewById(R.id.resultsContainer)
        tvResult = findViewById(R.id.tvResult)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvInferenceTime = findViewById(R.id.tvInferenceTime)
        tvAllProbs = findViewById(R.id.tvAllProbs)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)

        // Set model name
        tvModelName.text = when (modelType) {
            ModelManager.ModelType.GRAIN_CLASSIFICATION -> "Grain Classification"
            ModelManager.ModelType.DEFECT_CLASSIFICATION -> "Leaf Defect Detection"
        }
    }

    private fun setupClassifier() {
        try {
            val modelManager = MainActivity.instance.getModelManager()
            val interpreter = modelManager.getInterpreter(modelType)
            val labels = modelManager.getLabels(modelType)
            modelConfig = modelManager.getConfig(modelType)

            classifier = UnifiedClassifier(
                interpreter = interpreter,
                labels = labels,
                inputSize = modelConfig.inputSize,
                isDetection = modelConfig.isDetection
            )

        } catch (e: Exception) {
            Toast.makeText(this, "Failed to setup classifier: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun setupListeners() {
        btnBack.setOnClickListener {
            finish()
        }

        btnGallery.setOnClickListener {
            checkPermissionAndOpenGallery()
        }

        btnCamera.setOnClickListener {
            checkPermissionAndOpenCamera()
        }
    }

    private fun checkPermissionAndOpenGallery() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_MEDIA_IMAGES
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                openGallery()
            } else {
                requestPermission.launch(Manifest.permission.READ_MEDIA_IMAGES)
            }
        } else {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                openGallery()
            } else {
                requestPermission.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }
    }

    private fun checkPermissionAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            openCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openGallery() {
        pickImage.launch("image/*")
    }

    private fun openCamera() {
        takePhoto.launch(null)
    }

    private fun handleImageSelection(uri: Uri) {
        lifecycleScope.launch {
            try {
                showLoading(true)

                val bitmap = withContext(Dispatchers.IO) {
                    ImageUtils.getBitmapFromUri(this@AnalysisActivity, uri)
                }

                if (bitmap == null) {
                    Toast.makeText(
                        this@AnalysisActivity,
                        "Failed to load image",
                        Toast.LENGTH_SHORT
                    ).show()
                    showLoading(false)
                    return@launch
                }

                ivPreview.setImageBitmap(bitmap)
                runInference(bitmap)

            } catch (e: Exception) {
                showLoading(false)
                Toast.makeText(
                    this@AnalysisActivity,
                    "Error: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun handleCameraBitmap(bitmap: Bitmap) {
        ivPreview.setImageBitmap(bitmap)
        runInference(bitmap)
    }

    private fun runInference(bitmap: Bitmap) {
        lifecycleScope.launch {
            try {
                showLoading(true)
                resultsContainer.visibility = View.GONE

                val result = withContext(Dispatchers.IO) {
                    classifier.classify(bitmap)
                }

                displayResult(result)
                showLoading(false)

            } catch (e: Exception) {
                showLoading(false)
                Toast.makeText(
                    this@AnalysisActivity,
                    "Inference failed: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
                e.printStackTrace()
            }
        }
    }

    private fun displayResult(result: UnifiedClassifier.ClassificationResult) {
        resultsContainer.visibility = View.VISIBLE

        // Main result
        tvResult.text = result.label

        // Confidence
        val confidenceText = "Confidence: ${String.format("%.1f%%", result.confidence * 100)}"
        tvConfidence.text = confidenceText

        // Inference time
        tvInferenceTime.text = "Inference time: ${result.inferenceTime}ms"

        // Color code based on confidence
        val color = when {
            result.confidence > 0.9 -> ContextCompat.getColor(this, android.R.color.holo_green_dark)
            result.confidence > 0.7 -> ContextCompat.getColor(this, android.R.color.holo_orange_dark)
            else -> ContextCompat.getColor(this, android.R.color.holo_red_dark)
        }
        tvResult.setTextColor(color)

        // Show all probabilities
        result.allProbabilities?.let { probs ->
            val labels = MainActivity.instance.getModelManager().getLabels(modelType)
            val probsText = buildString {
                append("All probabilities:\n")
                probs.forEachIndexed { index, prob ->
                    if (index < labels.size) {
                        append("${labels[index]}: ${String.format("%.1f%%", prob * 100)}\n")
                    }
                }
            }
            tvAllProbs.text = probsText
            tvAllProbs.visibility = View.VISIBLE
        }
    }

    private fun showLoading(isLoading: Boolean) {
        progressBar.visibility = if (isLoading) View.VISIBLE else View.GONE
        btnCamera.isEnabled = !isLoading
        btnGallery.isEnabled = !isLoading
    }
}