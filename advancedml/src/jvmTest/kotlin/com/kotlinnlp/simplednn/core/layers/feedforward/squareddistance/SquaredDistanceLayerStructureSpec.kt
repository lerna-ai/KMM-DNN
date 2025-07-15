package com.kotlinnlp.simplednn.core.layers.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

class SquaredDistanceLayerStructureSpec : Spek({

  describe("a Square Distance Layer")
  {

    context("forward") {

      val layer = SquaredDistanceLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.5928f)),
            tolerance = 1.0e-05f)
        }
      }
    }

    context("backward") {

      val layer = SquaredDistanceLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(SquaredDistanceLayerUtils.getOutputErrors())
      val paramsErrors = layer.backward(propagateToInput = true)

      val params = layer.params

      it("should match the expected errors of the inputArray") {
        assertTrue {
          layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.9568f, -0.848f, 0.5936f)),
            tolerance = 1.0e-05f)
        }
      }

      it("should match the expected errors of the weights") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.wB)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.2976f, -0.496f, 0.3968f),
              floatArrayOf(0.0144f, 0.024f, -0.0192f),
              floatArrayOf(-0.1488f, -0.248f, 0.1984f),
              floatArrayOf(-0.1584f, -0.264f, 0.2112f),
              floatArrayOf(0.024f, 0.04f, -0.032f)
            )),
            tolerance = 1.0e-05f)
        }
      }
    }
  }
})
