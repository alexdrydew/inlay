# Inlay

Inlay is a static dependency injection library for Python that aims to make dependency injection both safe and expressive.

## Why inlay

Inlay makes it possible to decouple abstract syntax of your application from implementation leveraging Python structural types and static validation. Inlay is able to resolve complex types structurally from a configured registry safely and provide efficient runtime implementation for it.

## Goals

* Type safety: inlay will never lie about resolved object types. If no issues were found during dependency graph compilation resulting runtime object will strictly match the provided type annotation.
* Correctness: availability of all static and runtime dependencies is checked upfront, all dependencies are provided through type safe interface.
* Flexibility: inlay aims to infer complex types from simple registries so application type shape is further decoupled from implementation.

