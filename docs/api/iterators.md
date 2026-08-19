---
description: API reference for the ngio processing iterators.
---

# Iterators API reference

## ImageProcessingIterator

::: ngio.iterators.ImageProcessingIterator

## SegmentationIterator

::: ngio.iterators.SegmentationIterator

## MaskedSegmentationIterator

::: ngio.iterators.MaskedSegmentationIterator

## FeatureExtractorIterator

::: ngio.iterators.FeatureExtractorIterator

## ObjectDetectionIterator

::: ngio.iterators.ObjectDetectionIterator

## Configuration

### StitchConfig

::: ngio.iterators.StitchConfig

### NmsConfig

::: ngio.iterators.NmsConfig

## Mappers

### ThreadedMapper

::: ngio.iterators.ThreadedMapper

### ProcessMapper

::: ngio.iterators.ProcessMapper

### BasicMapper

::: ngio.iterators.BasicMapper

### MapperProtocol

::: ngio.iterators.MapperProtocol

### IterUnit

::: ngio.iterators.IterUnit

## AbstractIteratorBuilder

The shared method surface of every iterator — the reshaping calls
(`grid`, `by_chunks`, `product`, `with_halo`) and the execution calls
(`iter`, `map`, `reduce`).

::: ngio.iterators.AbstractIteratorBuilder
