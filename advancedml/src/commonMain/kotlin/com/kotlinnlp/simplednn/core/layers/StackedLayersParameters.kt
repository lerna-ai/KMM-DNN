/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * [StackedLayersParameters] contains all the parameters of the layers defined in [layersConfiguration],
 * grouped per layer.
 *
 * @property layersConfiguration a list of configurations, one per layer
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class StackedLayersParameters(
  val layersConfiguration: List<LayerInterface>,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [StackedLayersParameters] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [StackedLayersParameters]
     *
     * @return the [StackedLayersParameters] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): StackedLayersParameters = Serializer.deserialize(inputStream)
  }

  /**
   * Secondary constructor.
   *
   * @param layersConfiguration a list of configurations, one per layer
   * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   *
   * @return a new NeuralNetwork
   */
  constructor(
    vararg layersConfiguration: LayerInterface,
    weightsInitializer: Initializer? = GlorotInitializer(),
    biasesInitializer: Initializer? = GlorotInitializer()
  ): this(
    layersConfiguration = layersConfiguration.toList(),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )

  /**
   * The type of the input array.
   */
  val inputType: LayerType.Input = this.layersConfiguration.first().type

  /**
   * Whether the input array is sparse binary.
   */
  val sparseInput: Boolean = this.inputType == LayerType.Input.SparseBinary

  /**
   * The size of the input, meaningful when the first layer is not a Merge layer.
   */
  val inputSize: Int = this.layersConfiguration.first().size

  /**
   * The size of the inputs, meaningful when the first layer is a Merge layer.
   */
  val inputsSize: List<Int> = this.layersConfiguration.first().sizes

  /**
   * The output size.
   */
  val outputSize: Int = this.layersConfiguration.last().size

  /**
   * The [LayerParameters] of each layer.
   */
  val paramsPerLayer: List<LayerParameters> = List(this.layersConfiguration.size - 1) { i ->
    LayerParametersFactory(
      inputsSize = this.layersConfiguration[i].sizes,
      outputSize = this.layersConfiguration[i + 1].size,
      connectionType = this.layersConfiguration[i + 1].connectionType!!,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer,
      sparseInput = this.layersConfiguration[i].type == LayerType.Input.SparseBinary)
  }

  /**
   * The number of stacked layers.
   */
  val numOfLayers: Int = this.paramsPerLayer.size

  /**
   * Serialize this [StackedLayersParameters] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [StackedLayersParameters]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * @param index a stacked-layer index
   *
   * @return the parameters of the given layer, casted to the given type
   */
  inline fun <reified T : LayerParameters>getLayerParams(index: Int): T {

    require(index >= 0 && index < this.numOfLayers) { "Layer index ($index) out of range ([0, ${numOfLayers - 1}])." }

    return this.paramsPerLayer[index] as T
  }
}
