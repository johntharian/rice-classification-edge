package com.ricedetector

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var modelManager: ModelManager
    private lateinit var loadingContainer: View
    private lateinit var cardsContainer: LinearLayout
    private lateinit var cardGrainClassification: LinearLayout
    private lateinit var cardDefectClassification: LinearLayout
    private lateinit var cardGrainDetection: LinearLayout

    companion object {
        const val EXTRA_MODEL_TYPE = "model_type"
        lateinit var instance: MainActivity
            private set
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        instance = this

        // Initialize views
        loadingContainer = findViewById(R.id.loadingContainer)
        cardsContainer = findViewById(R.id.cardsContainer)
        cardGrainClassification = findViewById(R.id.cardGrainClassification)
        cardDefectClassification = findViewById(R.id.cardDefectClassification)

        // Setup click listeners
        setupClickListeners()

        // Initialize models
        initializeModels()
    }

    private fun setupClickListeners() {
        cardGrainClassification.setOnClickListener {
            openAnalysisScreen(ModelManager.ModelType.GRAIN_CLASSIFICATION)
        }

        cardDefectClassification.setOnClickListener {
            openAnalysisScreen(ModelManager.ModelType.DEFECT_CLASSIFICATION)
        }
    }

    private fun initializeModels() {
        lifecycleScope.launch {
            try {
                loadingContainer.visibility = View.VISIBLE
                cardsContainer.visibility = View.GONE

                withContext(Dispatchers.IO) {
                    modelManager = ModelManager(this@MainActivity)
                    modelManager.initializeAllModels()
                }

                loadingContainer.visibility = View.GONE
                cardsContainer.visibility = View.VISIBLE

                Toast.makeText(
                    this@MainActivity,
                    "All models loaded successfully!",
                    Toast.LENGTH_SHORT
                ).show()

            } catch (e: Exception) {
                loadingContainer.visibility = View.GONE
                Toast.makeText(
                    this@MainActivity,
                    "Failed to load models: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                e.printStackTrace()
            }
        }
    }

    private fun openAnalysisScreen(modelType: ModelManager.ModelType) {
        if (!::modelManager.isInitialized) {
            Toast.makeText(this, "Models not loaded yet", Toast.LENGTH_SHORT).show()
            return
        }

        val intent = Intent(this, AnalysisActivity::class.java)
        intent.putExtra(EXTRA_MODEL_TYPE, modelType.name)
        startActivity(intent)
    }

    fun getModelManager(): ModelManager {
        return modelManager
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::modelManager.isInitialized) {
            modelManager.close()
        }
    }
}