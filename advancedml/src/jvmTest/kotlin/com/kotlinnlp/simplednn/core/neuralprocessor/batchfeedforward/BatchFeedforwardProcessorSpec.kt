/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue


/**
 *
 */
class BatchFeedforwardProcessorSpec : Spek({

  describe("a BatchFeedforwardProcessor") {

    val inputSequence = BatchFeedforwardUtils.buildInputBatch()
    val model = BatchFeedforwardUtils.buildParams()
    val processor = BatchFeedforwardProcessor<DenseNDArray>(model = model, propagateToInput = true)
    val output = processor.forward(inputSequence)

    it("should match the expected first output array") {
      assertTrue {
        output[0].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.66959f, -0.793199f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected second output array") {
      assertTrue {
        output[1].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.739783f, 0.197375f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected third output array") {
      assertTrue {
        output[2].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.318521f, -0.591519f)),
          tolerance = 1.0e-06f
        )
      }
    }

    processor.backward(outputErrors = BatchFeedforwardUtils.buildOutputErrors())

    val paramsErrors = processor.getParamsErrors()

    val params = model.paramsPerLayer[0] as FeedforwardLayerParameters

    it("should match the expected errors of the biases") {
      assertTrue {
        paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.290168f, -0.659261f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of the weights") {
      assertTrue {
        (paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.259834f, 0.293235f, -0.283416f),
            floatArrayOf(-0.497742f, -0.197778f, 0.41039f)
          )),
          tolerance = 1.0e-06f
        )
      }
    }

    val inputErrors: List<DenseNDArray> = processor.getInputErrors()

    it("should match the expected errors of first input array") {
      assertTrue {
        inputErrors[0].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.329642f, 0.160346f, -0.415821f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of second input array") {
      assertTrue {
        inputErrors[1].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.221833f, -0.095071f, 0.316905f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of third input array") {
      assertTrue {
        inputErrors[2].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.216483f, 0.243231f, 0.12538f)),
          tolerance = 1.0e-06f
        )
      }
    }
  }
})
