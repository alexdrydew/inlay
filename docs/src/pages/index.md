---
title: Inlay
description: Static, structural dependency injection for Python
hide_table_of_contents: true
---

# Inlay

Inlay is a DI ([dependency injection](https://en.wikipedia.org/wiki/Dependency_injection)) library that introduces static and safe DI into the Python world.

Inlay primary interface is a `compile(Target, Registry) -> ResolvedTarget` function where
* `Target` is a supported python type annotation or a function stub;
* `ResolvedTarget` is an instance of annotated type or a function implementation;
* `Registry` is a configuration object with concrete "recipes" for resolving registered types.

The power of Inlay is in its supported variety of compilation targets. Beyond direct resolution of a registered type `T` instance against a requested dependency of type `K`, where type `K` is a supertype of type `K` (i.e. type `K` is an interface which type `T` satisfies), the following types are supported:

* functions and methods: Inlay can provide a function implementation just from its signature: return value is constructed from the available DI context enriched with the provided parameters. This makes it possible to express conditional dependencies and scopes, i.e. dependencies that are available only once runtime values are provided (usually in a form of function parameters);
* Protocols and TypedDicts, where injected object is assembled structurally from resolution of its individual members;
* union types, optionals;
* and more...

Importantly, `compile` strictly obeys the Python type system. It will either fail as early as possible (usually during module import) explaining why there is no way to assemble type safe implementation of the requested target under the provided registry, or provide implementation that will be fully type safe.

