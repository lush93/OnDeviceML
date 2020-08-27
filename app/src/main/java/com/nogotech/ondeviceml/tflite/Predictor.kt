package com.nogotech.ondeviceml.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.os.Trace
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import timber.log.Timber
import java.io.IOException
import java.nio.MappedByteBuffer
import java.util.*
import kotlin.Comparator
import kotlin.collections.ArrayList
import kotlin.math.min

/** A classifier specialized to label images using TensorFlow Lite. */
abstract class Predictor(activity: Activity, device: Device, numThreads: Int) {
    companion object{
        /**
         * Creates a classifier with the provided configuration.
         *
         * @param activity The current Activity.
         * @param model The model to use for classification.
         * @param device The device to use for classification.
         * @param numThreads The number of threads to use for classification.
         * @return A classifier with the desired configuration.
         */
        @Throws(IOException::class)
        fun create(activity: Activity, model: Model, device: Device, numThreads: Int): Predictor {
            return FloatEfficientNet(activity, device, numThreads)
        }

        /** An immutable result returned by a Classifier describing what was recognized.  */
        class Recognition(
            /**
             * A unique identifier for what has been recognized. Specific to the class, not the instance of
             * the object.
             */
            private val id: String,
            /** Display name for the recognition.  */
            private val title: String,
            /**
             * A sortable score for how good the recognition is relative to others. Higher should be better.
             */
            private val confidence: Float,
        ){
            fun getConfidence(): Float {
                return confidence
            }

            override fun toString(): String {
                val resultString = "[$id] $title " + String.format("(%.1f%%) ", confidence * 100.0f)

                return resultString.trim { it <= ' ' }
            }
        }

        /** Gets the results.  */
        private fun getResults(labelProb: Map<String, Float>): List<Recognition> {
            // Find the best classifications.
            val pq = PriorityQueue<Recognition>(
                1
            ){ lhs, rhs ->
                // Intentionally reversed to put high confidence at the head of the queue.
                rhs.getConfidence().compareTo(lhs.getConfidence())
            }

            for ((key, value) in labelProb) {
                pq.add(Recognition("" + key, key, value))
            }

            val recognitions = ArrayList<Recognition>()
            recognitions.add(pq.poll()!!)
            return recognitions
        }
    }

    /** The model type used for classification.  */
    enum class Model {
        FLOAT,
        QUANTIZED
    }

    /** The runtime device type used for executing classification.  */
    enum class Device {
        CPU,
        NNAPI,
        GPU
    }

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = FileUtil.loadMappedFile(activity, getModelPath())

    /** Image size along the x axis.  */
    private val imageSizeX: Int

    /** Image size along the y axis.  */
    private val imageSizeY: Int

    /** Optional GPU delegate for accleration.  */
    private var gpuDelegate: GpuDelegate? = null

    /** Optional NNAPI delegate for accleration.  */
    private var nnApiDelegate: NnApiDelegate? = null

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    private var tflite: Interpreter?

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    /** Labels corresponding to the output of the vision model.  */
    private var labels: List<String>

    /** Input image TensorBuffer.  */
    private var inputImageBuffer: TensorImage

    /** Output probability TensorBuffer.  */
    private var outputProbabilityBuffer: TensorBuffer

    /** Processer to apply post processing of the output probability.  */
    private var probabilityProcessor: TensorProcessor

    init{
        when (device) {
            Device.NNAPI -> {
                nnApiDelegate = NnApiDelegate()
                tfliteOptions.addDelegate(nnApiDelegate)
            }
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                tfliteOptions.addDelegate(gpuDelegate)
            }
            Device.CPU -> {
            }
        }
        tfliteOptions.setNumThreads(numThreads)
        tflite = Interpreter(tfliteModel!!, tfliteOptions)

        // Loads labels out from the label file.
        labels = FileUtil.loadLabels(activity, getLabelPath())

        // Reads type and shape of input and output tensors, respectively.
        val imageTensorIndex = 0
        val imageShape = tflite!!.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]
        val imageDataType = tflite!!.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape =
            tflite!!.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}
        val probabilityDataType = tflite!!.getOutputTensor(probabilityTensorIndex).dataType()

        // Creates the input tensor.
        inputImageBuffer = TensorImage(imageDataType)

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)

        // Creates the post processor for the output probability.
        probabilityProcessor = TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build()

        Timber.d("Created a Tensorflow Lite Image Classifier.")
    }

    /** Runs inference and returns the classification results.  */
    fun recognizeImage(bitmap: Bitmap, sensorOrientation: Int, resizeMethod: ResizeMethod): List<Recognition> {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")
        val startTimeForLoadImage = SystemClock.uptimeMillis()
        inputImageBuffer = loadImage(bitmap, sensorOrientation, resizeMethod)
        val endTimeForLoadImage = SystemClock.uptimeMillis()
        Trace.endSection()
        Timber.v("Timecost to load the image: %s", (endTimeForLoadImage - startTimeForLoadImage))

        // Runs the inference call.
        Trace.beginSection("runInference")
        val startTimeForReference = SystemClock.uptimeMillis()
        tflite?.run(inputImageBuffer.buffer, outputProbabilityBuffer.buffer.rewind())
        val endTimeForReference = SystemClock.uptimeMillis()
        Trace.endSection()
        Timber.v("Timecost to run model inference: %s", (endTimeForReference - startTimeForReference))

        // Gets the map of label and probability.
        val labeledProbability =
            TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                .mapWithFloatValue
        Trace.endSection()

        // Gets results.
        return getResults(labeledProbability)
    }

    /** Closes the interpreter and model to release resources.  */
    fun close() {
        tflite?.let{
            tflite?.close()
            tflite = null
        }
        gpuDelegate?.let{
            gpuDelegate?.close()
            gpuDelegate = null
        }
        nnApiDelegate?.let{
            nnApiDelegate?.close()
            nnApiDelegate = null
        }
        tfliteModel= null
    }

    /** Get the image size along the x axis.  */
    fun getImageSizeX(): Int {
        return imageSizeX
    }

    /** Get the image size along the y axis.  */
    fun getImageSizeY(): Int {
        return imageSizeY
    }

    /** Loads input image, and applies preprocessing.  */
    private fun loadImage(bitmap: Bitmap, sensorOrientation: Int, resizeMethod: ResizeMethod): TensorImage {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap)

        // Creates processor for the TensorImage.
        //val cropSize = min(bitmap.width, bitmap.height)
        val namRotation = sensorOrientation / 90
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        val imageProcessor = ImageProcessor.Builder()
            //.add(ResizeWithCropOrPadOp(imageSizeX, imageSizeY))
            .add(ResizeOp(imageSizeX, imageSizeY, resizeMethod))
            .add(Rot90Op(namRotation))
            .add(getPreprocessNormalizeOp())
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    /** Gets the name of the model file stored in Assets.  */
    protected abstract fun getModelPath(): String

    /** Gets the name of the label file stored in Assets.  */
    protected abstract fun getLabelPath(): String

    /** Gets the TensorOperator to normalize the input image in preprocessing.  */
    protected abstract fun getPreprocessNormalizeOp(): TensorOperator

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     *
     * For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract fun getPostprocessNormalizeOp(): TensorOperator
}