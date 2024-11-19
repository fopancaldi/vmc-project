Theoretical and Numerical Aspects of Nuclear Physics project by Lorenzo Fabbri and Francesco Orso Pancaldi.

# Coding conventions

1. Variables and constants start with lowercase, classes and functions with uppercase.
2. Use mixed case instead of underscore to separate words.
3. Variable names should be short but expressive. The length of a name should be proportional to the lifetime of the variable.
4. Be (very) generous with `const` and `assert`. Actually, put `const` everywhere (inlcuding functions!) except where it causes an error.
5. No non-`constexpr` global variables.
6. Test **everything**.
7. In for-loops, avoid indices at all costs (unless you have to count something). Prefer range-based loops and/or algorithms.
8. (Debatable) even though initializing with `{}` is safer, I'd rather use `=` since it's nicer.
9. Use `int` (or better, `IntType`) when an integer type is needed, even if the integer is guaranteed to be positive. Use `unsigned int` (or `UIntType`, or `std::size_t` in the unlikely case when there are overflow concerns) only to deal with sizes (example: initialize an `std::array`).
10. Use `!=` instead of `<` or `<=` in for-loops. It communicates to the reader that you are guaranteeing that the index will always be less than the max and will never jump (i.e. increase by more than 1 or similar).
11. No magic constants, use `constexpr` instead.
12. Use template argument deduction whenever possible.
13. In loops where it is necessary to use indices, use **only** `IntType` or `UIntType` as the index type.
14. Use `i, j, ...` for `IntType` indices in loops, and `u, v, ...` for `UIntType` ones. Also do not initialize an `UInt` object with `0`!
15. Use asserts to handle unplanned inputs, not exceptions. If someone wants to compute the average of -10 numbers, the program should collapse. So, avoid exceptins altogether.
16. As always, object of primitive type should be passed by value and ones of derived type should be passed by `const &`. However:
    - `VarParams` objects are usually arrays of few `FPType` &rarr; pass by **value**.
    - `Bounds` objects also contain a few `FPType` &rarr; pass by **value**.
    - `RandomGenerator` objects must always be passed by **non-const reference**.
17.  Do **not** use `[=]` and `[&]` in the lambdas, I find it dangerous.
18.  Initialize every `FPType` object with a `float` or an `int`, they will be promoted.

# Coding remarks

1. All commits must have the necessary tests (and pass them, duh).
2. Documenting the code with Doxygen comments is the last thing to do. But short comments are always welcome.
3. Ideally, one puts the implementation details in a `.cpp`, and the declarations of the interface functions in a `.hpp`. But with templates, we are forced to put everything in `.hpp`. So I suggest the following convention: the functions that would usually be put in a `.cpp` file (i.e. the ones that we do not want the user to call, like `MetropolisUpdate`, which is just a cog in the machine) will have a trailing underscore.

# General remarks

1. There is an error message while building the container:
    > Error: there is no registered task type 'cppbuild'. Did you miss installing an extension that provides a corresponding task provider?
It is completely harmless.
See the related [issue](https://github.com/microsoft/vscode-cpptools/issues/6450).
2. We are only working in cartesian coordinates, at least for now.

# TODO:

1. Find a better way of estimating the constexpr constants, like `thermalizationMoves`.
2. `int` -> `float` is NOT a promotion! Change the program accordingly. Actually, it is a numeric conversion, and in particular a *safe* conversion, if the integer is reasonably small.
3. There are too many `.val`. Define opportune operators (for example, `operator+` for the masses).