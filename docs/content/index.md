---
title: Inlay
description: Typed dependency contexts for Python.
sidebar_position: 1
hide_table_of_contents: true
---

# Inlay

Inlay is a Python library for building typed hierarchical dependency contexts.

## Why use Inlay

:::note[TL;DR]

Inlay is for code where “what dependencies are available?” depends on where you are in the program.

It lets you describe that availability as a typed Python context, progressively derive richer contexts from runtime values, and validate the resulting dependency graph before use.

:::

Code needs dependencies to do useful work: clients, configurations, runtime values, and many other things. As a program grows in complexity, dependencies stop being universally available and become tied to the execution contexts: while a single HTTP request data can only be available only inside the corresponding endpoint handler many dependencies are shared by a server process. That creates a problem of referencing correct dependencies at the currently executing line of code (which is often buried in the call stack).

This problem is well studied, and there are already many ways to solve it: function arguments, dependency inversion, DI containers. Inlay suggests another approach: declare the *context* explicitly as a Python type, such as [`Protocol`](https://typing.python.org/en/latest/spec/protocol.html) (or [`TypedDict`](https://typing.python.org/en/latest/spec/typeddict.html)). Inlay can then construct and validate that context as a dependency graph. Because contexts can have methods that return other contexts, they can describe a whole hierarchy of dependency availability, which can efficiently model progressively extending set of dependencies available to a program during its lifecycle.

