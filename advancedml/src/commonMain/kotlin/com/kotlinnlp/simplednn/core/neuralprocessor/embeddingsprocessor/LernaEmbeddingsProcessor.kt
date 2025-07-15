/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.embeddings.LernaEmbeddingsMap
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import korlibs.io.lang.assert
import korlibs.io.lang.format

/**
 * The neural processor that manages embeddings in a map.
 *
 * @param embeddingsMap an embeddings map
 * @param dropout the probability to get the unknown embedding (default 0.0)
 */
open class LernaEmbeddingsProcessor<T>(
  private val embeddingsMap: LernaEmbeddingsMap<T>,
  private val dropout: Float = 0.0f
) : NeuralProcessor<
        List<T>, // InputType
        List<DenseNDArray>, // OutputType
        List<DenseNDArray>, // ErrorsType
        NeuralProcessor.NoInputErrors // InputErrorsType
        > {

  /**
   * Whether to propagate the errors to the input during the backward (not supported)
   */
  override val propagateToInput: Boolean = false

  /**
   * The id for the pool (not supported).
   */
  override val id: Int = 0

  /**
   * List of embeddings used during the last forward.
   */
  private var usedEmbeddings = listOf<ParamsArray>()

  private var embeddingGradients = mutableListOf<Float>()

  private var lenMultiHotEmbeddings: List<Int> = listOf()
  /**
   * List of embeddings errors resulting from the last backward.
   */
  private val errorsAccumulator by lazy { ParamsErrorsAccumulator() }

  /**
   * Check the dropout value.
   */
  init {
    require(this.dropout in 0.0 .. 1.0) { "The dropout probability must be in the range [0.0, 1.0]."}
  }

  /**
   * Execute the forward of the input to the output.
   *
   * @param input the embeddings keys
   *
   * @return the embeddings vectors associated to the given keys
   */
  override fun forward(input: List<T>): List<DenseNDArray> {
    this.usedEmbeddings = input.map { this.embeddingsMap.get(it as Int, this.dropout) }
    return this.usedEmbeddings.map { it.values }
  }

//  fun forward(categoricalFeature: MutableList<Int>, numericalFeature: MutableMap<Int, Float>, multiHotFeature: MutableMap<Int, Int>): List<DenseNDArray>
//  {
//    val categoricalEmbedding = categoricalFeature.map {this.embeddingsMap.get(it, this.dropout).values }
//    val numericalEmbedding = mutableListOf<DenseNDArray>()
//    numericalFeature.forEach { (key, value) -> numericalEmbedding.add(this.embeddingsMap.get(key, this.dropout).values.prod(value)) }
//
//    // total possible values of a multi-hot feature, doesn't matter if they are present in the current example or not
//    val totalMultiHot = multiHotFeature.size.toFloat()
//
//    val multiHotEmbedding = mutableListOf<DenseNDArray>()
//    multiHotFeature.forEach { (key, value) -> if (value !=0) {multiHotEmbedding.add(this.embeddingsMap.get(key, this.dropout).values)} }
//
//    val multiHot = multiHotEmbedding.reduce{ acc, dense_array -> acc.assignSum(dense_array) }
//    multiHot.assignDiv(totalMultiHot)
//
//    val embedding = categoricalEmbedding + numericalEmbedding + mutableListOf(multiHot)
//    val usedKeys = categoricalFeature + numericalFeature.keys + multiHotFeature.filterValues{ it != 0 }.keys
//
//    // values of multi-hot feature which are 1 in current example, i.e contribute to the embedding
//    this.numMultiHotEmbeddings = multiHotFeature.values.filter { it != 0 }.size
//
//    this.usedEmbeddings = usedKeys.map{this.embeddingsMap.get(it, this.dropout)}
//
//    categoricalFeature.forEach { this.embeddingGradients.add(1.0) }
//    numericalFeature.forEach { (_, value) -> this.embeddingGradients.add(value) }
//    multiHotFeature.forEach { (_, value) -> if (value != 0) {this.embeddingGradients.add(1.0/totalMultiHot) } }
//
////        println("gradients are ${this.embeddingGradients}")
////        println("used embeddings are:")
////        this.usedEmbeddings.forEach{ println(it.values) }
//
//    return embedding
//  }

  fun forward(
    categoricalFeature: MutableList<Int>? = null,
    numericalFeature: MutableMap<Int, Float>? = null,
    multiHotFeatures: List<MutableMap<Int, Int>>? = null
  ): List<Any> {

    // List<ParamArray>: where each ParamArray is the embedding for a single categorical variable
    val categoricalEmbedding: List<ParamsArray> = categoricalFeature?.map { this.embeddingsMap.get(it, this.dropout) } ?: emptyList()
    val numericalFeatureValues = mutableListOf<Float>()
    val numericalEmbedding: List<ParamsArray> = numericalFeature?.map { (key, value) ->
      val embedding = this.embeddingsMap.get(key, this.dropout)
      numericalFeatureValues.add(value)
        embedding
    } ?: emptyList()

    // List<List<ParamArray>>: where the inner List<DenseNDArray> is the list of embeddings for each possible value of a single multi-hot variable
    // The outer list is the list across all multi-hot features
    val multiHotList: List<List<ParamsArray>> = multiHotFeatures?.map { multiHotFeature ->
      multiHotFeature.filterValues { it != 0 }.map { (key, _) ->
        this.embeddingsMap.get(key, this.dropout)
      }
    } ?: emptyList()


    // Total possible values (1/present or 0/absent) of a multi-hot variable, doesn't matter if they are present in the current example or not
    // this value is used to normalize the embedding of a multi-hot variable

    val totalMultiHot: List<Float> = multiHotFeatures?.map { it.size.toFloat() } ?: emptyList()

    // The final embedding of a given multi-hot variable is obtained by the mean of all possible true values (i.e. where value in the MutableMap != 0)
    val multiHotEmbeddings: List<DenseNDArray> = multiHotList.mapIndexed{index, param_embedding_list -> param_embedding_list.map{it.values}
      .reduce{accumulator, dense_array -> accumulator.assignSum(dense_array)}.assignDiv(
        totalMultiHot[index]
      )}

    val embedding = categoricalEmbedding.map { it.values } +
            numericalEmbedding.mapIndexed{index, embedding_array -> embedding_array.values.prod(
              numericalFeatureValues[index]
            )} +
            multiHotEmbeddings

    this.lenMultiHotEmbeddings = multiHotFeatures?.map { feature_dict -> feature_dict.filterValues{ it != 0 }.size } ?: emptyList()

    this.usedEmbeddings = categoricalEmbedding + numericalEmbedding + multiHotList.flatten()

    val usedKeys = (categoricalFeature ?: emptyList()) + (numericalFeature?.keys ?: emptyList()) + (multiHotFeatures?.flatMap { it.filterValues { it != 0 }.keys }
      ?: emptyList())

    assert(this.usedEmbeddings.size == usedKeys.size) {
      "Number of used embeddings (${this.usedEmbeddings.size}) does not match the number of used keys (${usedKeys.size})"
    }

    categoricalFeature?.forEach { this.embeddingGradients.add(1.0f) }
    numericalFeature?.forEach { (_, value) -> this.embeddingGradients.add(value) }
    multiHotFeatures?.forEachIndexed { index, multiHotFeature ->
      multiHotFeature.forEach { (_, value) ->
        if (value != 0) {
          this.embeddingGradients.add(1.0f / totalMultiHot[index])
        }
      }
    }

    return embedding
  }



  /**
   * Accumulate errors into the last given embeddings
   *
   * @param outputErrors the gradient/errors of the output wrt to its previous layers
   */
//  override fun backward(outputErrors: List<DenseNDArray>) {
//
//    // repeat last error for each categorical feature upto n times, get n from class variable
//
//    val embeddingErrors = mutableListOf<DenseNDArray>()
//    embeddingErrors.addAll(outputErrors)
//    (0 until this.numMultiHotEmbeddings - 1).forEach { embeddingErrors.add(outputErrors.last()) }
//
//    val gradient = embeddingErrors.mapIndexed{index, error -> error.prod(this.embeddingGradients[index])}
//
//    require(gradient.size == this.usedEmbeddings.size) {
//      "Number of errors (%d) does not reflect the number of used embeddings (%d)".format(
//        gradient.size, this.usedEmbeddings.size)
//    }
//
//    this.errorsAccumulator.clear()
//
//    this.usedEmbeddings.zip(gradient).forEach { (embedding, errors) ->
//      this.errorsAccumulator.accumulate(embedding, errors)
//    }
//  }

    override fun backward(outputErrors: List<DenseNDArray>) {

    val embeddingErrors = mutableListOf<DenseNDArray>()

    // get how many multi-hot feature embeddings are there in the current example (let's say n)
    // the last 'n' errors of the outputErrors belong to the multi-hot embeddings
    // they need to be repeated to match the length of number of features involved in computing each
    // multi-hot embedding
    val numMultiHotEmbeddings = this.lenMultiHotEmbeddings.size

    val multiHotErrors = outputErrors.takeLast(numMultiHotEmbeddings)

    embeddingErrors.addAll(outputErrors.dropLast(numMultiHotEmbeddings))

    // repeat the global error for each multi-hot variable 'k' times where k is the number of features involved in computing that multi-hot embedding
    this.lenMultiHotEmbeddings.forEachIndexed{ index, len_multi_hot_feat ->
      (0 until len_multi_hot_feat).forEach {
      embeddingErrors.add(multiHotErrors[index]) } }

    val gradient = embeddingErrors.mapIndexed{index, error -> error.prod(this.embeddingGradients[index])}

    require(gradient.size == this.usedEmbeddings.size) {
      "Number of errors (%d) does not reflect the number of used embeddings (%d)".format(
        gradient.size, this.usedEmbeddings.size)
    }

    this.errorsAccumulator.clear()

    this.usedEmbeddings.zip(gradient).forEach { (embedding, errors) ->
      this.errorsAccumulator.accumulate(embedding, errors)
    }
  }

  /**
   * No input errors available.
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors

  /**
   * Return the embeddings errors accumulated with the last backward.
   *
   * @param copy whether the returned errors must be a copy or a reference (default true)
   *
   * @return the accumulated errors of the last used embeddings
   */
  override fun getParamsErrors(copy: Boolean) = this.errorsAccumulator.getParamsErrors(copy = copy)
}
