Theoretical and Numerical Aspects of Nuclear Physics project by Lorenzo Fabbri and Francesco Orso Pancaldi.

# Coding conventions

1. Variables and constants start with lowercase, classes and functions with uppercase.
2. Use mixed case instead of underscore to separate words.
3. Variable names should be short but expressive. The length of a name should be proportional to the lifetime of the variable.
4. Be (very) generous with `const` and `assert`. Actually, put `const` everywhere (inlcuding functions!) except where it causes an error.
5. No global variables.
6. Test **everything**.
7. In for-loops, avoid indices at all costs. Prefer range-based loops and/or algorithms.

# Coding remarks

1. All commits must have the necessary tests (and pass them, duh).
2. Documenting the code with Doxygen comments is the last thing to do. But short comments are always welcome.

# General remarks

There is an error message while building the container:

> Error: there is no registered task type 'cppbuild'. Did you miss installing an extension that provides a corresponding task provider?
 
It is completely harmless.
See the related [issue](https://github.com/microsoft/vscode-cpptools/issues/6450).