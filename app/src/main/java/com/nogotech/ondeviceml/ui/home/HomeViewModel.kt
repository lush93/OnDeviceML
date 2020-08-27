package com.nogotech.ondeviceml.ui.home

import android.app.Application
import android.graphics.Bitmap
import android.os.SystemClock
import androidx.lifecycle.*
import com.bumptech.glide.RequestManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import timber.log.Timber

class HomeViewModel(private val glide: RequestManager,
                    application: Application) : AndroidViewModel(application) {

    private lateinit var rgbFrameBitmap : Bitmap
    private var lateAssessment = false

    fun loadBitmap(resourceId: Int){
        viewModelScope.launch {
            val startTime = SystemClock.uptimeMillis()

            if (::rgbFrameBitmap.isInitialized) rgbFrameBitmap.recycle()
            rgbFrameBitmap = loadBitmapSuspend(resourceId)
            val lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            Timber.v("Time cost to load Image Bitmap: %s", lastProcessingTimeMs)

            if(lateAssessment){
                //val results = predict()
                //_recognitions.postValue(results)
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