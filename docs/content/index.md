---
title: Inlay
description: Typed dependency contexts for Python.
sidebar_position: 1
hide_table_of_contents: true
---

# Inlay

Inlay is a Python library for building typed hierarchical dependency contexts.

## Why use Inlay

Modern Python DI containers are usually type-aware and can provide individually requested typed dependencies. The limiting factor is that DI containers usually force an additional constraint on the app in one of the following forms:

* dependencies are injected at the entry point and then passed through the call stack, which still leads to manual dependency propagation boilerplate;
* alternatively DI container can be made directly accessible from inside the call stack, which makes dependency resolution inherently unsafe and makes dependencies unclear (which is also known as service locator anti-pattern).

Typed contexts solve both of these problems: global container is replaced with an explicit typed context that exactly describes all the required dependencies that must be available upfront. Inlay encourages using Python structural types (either [Protocol](https://typing.python.org/en/latest/spec/protocol.html) or [TypedDict](https://typing.python.org/en/latest/spec/typeddict.html)) for this, so that dependency context can be threaded through the call stack without polluting interfaces with unrelated dependencies.


