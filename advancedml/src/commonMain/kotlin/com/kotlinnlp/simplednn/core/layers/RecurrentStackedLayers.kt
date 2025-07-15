/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A structure of stacked layers used in recurrent networks, in which the output array of a layer references the input
 * array of the following.
 * This permits to optimize the forward and backward operations without duplicating values.
 *
 * @property params the parameters
 * @param dropouts the probability of dropout for each stacked layer
 * @property statesWindow the context window to get the previous and the next state of the structure
 */
internal class RecurrentStackedLayers<InputNDArrayType : NDArray<InputNDArrayType>>(
  params: StackedLayersParameters,
  dropouts: List<Double>,
  val statesWindow: StatesWindow<InputNDArrayType>
) : LayersWindow, StackedLayers<InputNDArrayType>(
  params = params,
  dropouts = dropouts
) {

  /**
   * A structure of stacked layers used in recurrent networks, in which the output array of a layer references the input
   * array of the following.
   * This permits to optimize the forward and backward operations without duplicating values.
   *
   * @param params the parameters
   * @param dropout the probability of dropout for each stacked layer
   * @param statesWindow the context window to get the previous and the next state of the structure
   */
  constructor(
    params: StackedLayersParameters,
    dropout: Double,
    statesWindow: StatesWindow<InputNDArrayType>
  ): this(
    params = params,
    dropouts = List(params.numOfLayers) { dropout },
    statesWindow = statesWindow
  )

  /**
   * A list of booleans indicating if the init hidden layers must be used in the next forward.
   */
  private var useInitHidden: List<Boolean> = this.layers.map { false }

  /**
   * The initial hidden layers from which to take the previous hidden if the method [setInitHidden] is called before a
   * forward.
   */
  private val initHiddenLayers: List<Layer<*>> = this.buildLayers(dropouts)

  /**
   * Set the initial hidden arrays of each layer. They will be used as previous hidden in the next forward.
   * Set [arrays] to null to don't use the initial hidden layers.
   *
   * @param arrays the list of initial hidden arrays (one per layer, can be null)
   */
  fun setInitHidden(arrays: List<DenseNDArray?>?) {
    require(arrays == null || arrays.size == this.layers.size) {
      "Incompatible init hidden arrays size (%d != %d).".format(arrays!!.size, this.layers.size)
    }

    if (arrays != null) {
      this.initHiddenLayers.zip(arrays).forEach { (layer, array) ->
        if (layer is RecurrentLayer && array != null) layer.setInitHidden(array)
      }
    }

    this.useInitHidden = arrays?.map { it != null } ?: this.layers.map { false }
  }

  /**
   * Get the errors of the initial hidden arrays.
   * This method should be used only if initial hidden arrays has been set with the [setInitHidden] method.
   *
   * @return the errors of the initial hidden arrays (null if no init hidden is used for a certain layer)
   */
  fun getInitHiddenErrors(): List<DenseNDArray?> =
    this.useInitHidden.zip(this.initHiddenLayers).map { (useInitHidden, layer) ->
      if (useInitHidden && layer is RecurrentLayer) layer.getInitHiddenErrors() else null
    }

  /**
   * @return the current layer in previous state
   */
  override fun getPrevState(): Layer<*>? = if (this.useInitHidden[this.curLayerIndex])
    this.initHiddenLayers[this.curLayerIndex]
  else
    this.statesWindow.getPrevState()?.layers?.get(this.curLayerIndex)

  /**
   * @return the current layer in next state
   */
  override fun getNextState(): Layer<*>? = this.statesWindow.getNextState()?.layers?.get(this.curLayerIndex)
}
