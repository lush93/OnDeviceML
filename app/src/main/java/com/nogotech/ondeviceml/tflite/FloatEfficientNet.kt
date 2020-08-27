package com.nogotech.ondeviceml.tflite

import android.app.Activity
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp

/** This TensorFlowLite classifier works with the float MobileNet model.  */
/**
 * Initializes a `ClassifierFloatMobileNet`.
 *
 * @param activity
 */
class FloatEfficientNet(activity: Activity,
                         device: Device,
                         numThreads: Int) : Predictor(activity, device, numThreads){
    companion object{
        /** Float MobileNet requires additional normalization of the used input.  */
        private const val IMAGE_MEAN = 127.5f

        private const val IMAGE_STD = 127.5f

        /**
         * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
         * and 1.0f, respectively, to bypass the normalization.
         */
        private const val PROBABILITY_MEAN = 0.0f

        private const val PROBABILITY_STD = 1.0f
    }

    override fun getModelPath(): String {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        return "efficientnet_lite0_fp32_2.tflite"
    }

    override fun getLabelPath(): String {
        return "labels_without_background.txt"
    }

    override fun getPreprocessNormalizeOp(): TensorOperator {
        return NormalizeOp(IMAGE_MEAN, IMAGE_STD)
    }

    override fun getPostprocessNormalizeOp(): TensorOperator {
        return NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)
    }
}