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

# Coding remarks

1. All commits must have the necessary tests (and pass them, duh).
2. Documenting the code with Doxygen comments is the last thing to do. But short comments are always welcome.

# General remarks

There is an error message while building the container:

> Error: there is no registered task type 'cppbuild'. Did you miss installing an extension that provides a corresponding task provider?
 
It is completely harmless.
See the related [issue](https://github.com/microsoft/vscode-cpptools/issues/6450).