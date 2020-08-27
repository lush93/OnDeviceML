package com.nogotech.ondeviceml

import android.app.Application
import timber.log.Timber

class OnDeviceApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        Timber.plant(Timber.DebugTree())
    }
}