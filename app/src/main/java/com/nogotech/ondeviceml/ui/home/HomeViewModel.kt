package com.nogotech.ondeviceml.ui.home

import android.app.Activity
import android.app.Application
import android.graphics.Bitmap
import android.os.SystemClock
import androidx.lifecycle.*
import com.bumptech.glide.RequestManager
import com.nogotech.ondeviceml.tflite.Predictor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.image.ops.ResizeOp
import timber.log.Timber
import java.io.IOException

class HomeViewModel(private val glide: RequestManager,
                    application: Application) : AndroidViewModel(application) {

    private lateinit var rgbFrameBitmap : Bitmap
    private var predictor : Predictor? = null
    private var lateAssessment = false

    private val _prediction = MutableLiveData<String>()
    val prediction get() = _prediction

    fun processImage(activity: Activity) {
        viewModelScope.launch {
            recreateClassifier(activity, Predictor.Model.FLOAT, Predictor.Device.CPU, 1)
            if (predictor == null) {
                Timber.e("No classifier on preview!")
                return@launch
            }
            if (!::rgbFrameBitmap.isInitialized) lateAssessment = true
            else {
                val results = predict()

                _prediction.postValue(results!![0].toString())
            }
        }
    }

    private suspend fun predict(): List<Predictor.Companion.Recognition>?{
        return withContext(Dispatchers.Default) {
            predictor?.let {
                val startTime = SystemClock.uptimeMillis()

                val results = predictor?.recognizeImage(rgbFrameBitmap, 0, ResizeOp.ResizeMethod.BILINEAR)

                val lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                Timber.v("Detect: %s", results!!)
                Timber.v("Time cost to Predict: %s", lastProcessingTimeMs)
                return@withContext results
            }
        }
    }

    private suspend fun recreateClassifier(activity : Activity, model: Predictor.Model, device: Predictor.Device, numThreads: Int) {
        withContext(Dispatchers.IO){
            predictor?.let {
                Timber.d("Closing classifier.")
                predictor?.close()
                predictor = null
            }
            if (device === Predictor.Device.GPU && model === Predictor.Model.QUANTIZED) {
                Timber.d("Not creating classifier: GPU doesn't support quantized models.")

                return@withContext
            }
            try {
                Timber.d(
                    "Creating classifier (model=%s, device=%s, numThreads=%d)".format(model, device, numThreads))
                predictor = Predictor.create(activity, model, device, numThreads)
            } catch (e: IOException) {
                Timber.e(e, "Failed to create classifier.")
            }
        }
    }


    fun loadBitmap(resourceId: Int){
        viewModelScope.launch {
            val startTime = SystemClock.uptimeMillis()

            if (::rgbFrameBitmap.isInitialized) rgbFrameBitmap.recycle()
            rgbFrameBitmap = loadBitmapSuspend(resourceId)
            val lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            Timber.v("Time cost to load Image Bitmap: %s", lastProcessingTimeMs)

            if(lateAssessment){
                val results = predict()

                _prediction.postValue(results!![0].toString())
            }
        }
    }

    private suspend fun loadBitmapSuspend(resourceId: Int) : Bitmap{
        return withContext(Dispatchers.IO){
            val futureTarget = glide
                .asBitmap()
                .load(resourceId)
                .submit()

            futureTarget.get()
        }
    }
}