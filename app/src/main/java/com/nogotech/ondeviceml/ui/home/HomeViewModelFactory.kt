package com.nogotech.ondeviceml.ui.home

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.bumptech.glide.RequestManager

class HomeViewModelFactory(
    private val glide: RequestManager,
    private val application: Application
) : ViewModelProvider.Factory {
    @Suppress("unchecked_cast")
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(HomeViewModel::class.java)) {
            return HomeViewModel(glide, application) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}