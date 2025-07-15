/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.utils



/**
 * A dictionary containing a set of elements.
 * Elements are mapped bi-univocally to ids.
 * It provides methods to get information about elements, like their occurrences count and frequency.
 */
class DictionarySet<T> {

  /**
   * A [DictionarySet] factory.
   */
  companion object Factory {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Build a [DictionarySet] containing the given [elements].
     *
     * @param elements the elements to insert into the building dictionary
     *
     * @return a new dictionary set containing the given [elements]
     */
    operator fun <T> invoke(elements: List<T>): DictionarySet<T> {

      val dictionary = DictionarySet<T>()

      elements.forEach { dictionary.add(it) }

      return dictionary
    }

    /**
     * Build a [DictionarySet] containing the given [elements].
     *
     * @param elements the elements to insert into the building dictionary
     *
     * @return a new dictionary set containing the given [elements]
     */
    operator fun <T> invoke(elements: Sequence<T>): DictionarySet<T> {

      val dictionary = DictionarySet<T>()

      elements.forEach { dictionary.add(it) }

      return dictionary
    }
  }

  /**
   * The number of distinct element of this set.
   */
  val size: Int get() = this.elementsMultiset.toSet().size

  /**
   * The elements multiset with adding properties (e.g. the count of insertions of an element).
   */
  private val elementsMultiset: HashSet<T> = hashSetOf()

  /**
   * The [BiMap] of elements to ids.
   */
  private val elementsBiMap: MutableBiMap<Any, Int> = HashBiMap()

  /**
   * @param element an element
   *
   * @return a Boolean indicating if the dictionary contains the given element
   */
  operator fun contains(element: T): Boolean = this.elementsBiMap.containsKey(element as Any)

  /**
   * Add the given [element] to the dictionary, incrementing the count of its occurrences.
   *
   * @param element the element to add
   *
   * @return the ID of the given [element]
   */
  fun add(element: T): Int {

    this.elementsMultiset.add(element)

    if (element as Any !in this.elementsBiMap) {
      this.elementsBiMap[element] = this.elementsMultiset.toSet().size - 1
    }

    return this.getId(element)!!
  }

  /**
   * Get the element associated to the given [id] if it exists, null otherwise.
   *
   * @param id the id of an element
   *
   * @return the element with the given [id] or null
   */
  fun getElement(id: Int): Any? = this.elementsBiMap.inverse[id]

  /**
   * @param element an element
   *
   * @return the id of the given [element] if it is present in the dictionary, null otherwise
   */
  fun getId(element: T): Int? = this.elementsBiMap[element as Any]

  /**
   * @param element an element
   *
   * @return the occurrences count of the given [element] (0 if it is not present)
   */
  fun getCount(element: T): Int = this.elementsMultiset.count{ it?.equals(element) ?: false }

  /**
   * @param id the id of an element
   *
   * @return the occurrences count of the element with the given [id] (0 if it is not present)
   */
  fun getCount(id: Int): Int = this.elementsMultiset.count{ it?.equals(this.getElement(id)) ?: false }

  /**
   * @param id the id of an element
   *
   * @return the count of the element with the given [id]  (0 if it is not present)
   */
  fun getFrequency(id: Int): Int = this.elementsMultiset.toList().count{ it?.equals(this.getElement(id)) ?: false } / this.size

  /**
   * @return a list of the elements in the dictionary
   */
  fun getElements(): List<T> = this.elementsMultiset.toSet().toList()

  /**
   * @return a set of the elements in the dictionary, sorted by ascending order of occurrences
   */
  fun getElementsSortedSet(): Set<T> =
    this.elementsMultiset.toSet().sortedBy { this.getCount(it) }.toSet()

  /**
   * @return a set of the elements in the dictionary, sorted by descending order of occurrences
   */
  fun getElementsReversedSet(): Set<T> =
    this.elementsMultiset.toSet().sortedByDescending { this.getCount(it) }.toSet()
}
