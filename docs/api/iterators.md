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

### BatchedMapper

::: ngio.iterators.BatchedMapper

### BasicMapper

::: ngio.iterators.BasicMapper

### MapperProtocol

::: ngio.iterators.MapperProtocol

### IterUnit

::: ngio.iterators.IterUnit

## Scheduling

The primitives behind the mappers' conflict-free schedule — useful for
inspecting how a tiling will parallelize or split into jobs.

### plan_waves

::: ngio.iterators.plan_waves

### canonical_unit_order

::: ngio.iterators.canonical_unit_order

### write_conflict_components

::: ngio.iterators.write_conflict_components

### compute_write_footprint

::: ngio.iterators.compute_write_footprint

## Types

### JobArgs

::: ngio.iterators.JobArgs

### TailPolicy

::: ngio.iterators.TailPolicy

### HaloMargins

::: ngio.iterators.HaloMargins

### FeatureFuncResult

::: ngio.iterators.FeatureFuncResult

### MaxWorkers

::: ngio.iterators.MaxWorkers

## AbstractIteratorBuilder

The shared method surface of every iterator — the reshaping calls
(`by_grid`, `by_blocks`, `by_chunks`, `by_write_units`, `product`, `with_halo`,
`for_job`), the execution calls (`iter`, `iter_batched`, `map`, `reduce` and
each iterator's terminal verb — `measure`, `detect`), and the
distributed-run init step (`prepare_jobs`).

::: ngio.iterators.AbstractIteratorBuilder
