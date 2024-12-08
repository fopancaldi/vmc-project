[Theoretical and Numerical Aspects of Nuclear Physics](https://www.unibo.it/en/study/phd-professional-masters-specialisation-schools-and-other-programmes/course-unit-catalogue/course-unit/2023/433587) project by Lorenzo Fabbri and Francesco Orso Pancaldi.

Follows the instructions of Morten Hjorth-Jensen's [project](https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/Projects/2023/Project1/pdf/Project1.pdf).

# Dependencies and Dev container

- To run the program: `tbb` (since the C++ `atomic` header requires it)
- To generate documentation: `doxygen`, `graphviz`

The program has a [Dev container](https://code.visualstudio.com/docs/devcontainers/containers) configuration file, so if the container is created (for example from VS Code) all required dependencies are automatically installed.
As the code was developed and tested inside a container, using one also guarantees the absence of architecture- or OS-specific bugs.
To run a Dev container, follow the [guide](https://code.visualstudio.com/docs/devcontainers/tutorial) or, in short:
1. Install [Docker](https://www.docker.com/).
2. Install the VS Code Dev container [extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
3. Either build the container from the window in the bottom-right that appears when VS Code is launched, or, from the top bar in VS Code, type `> Dev Containers: Build and Open in a Container`.

# Usage (Linux, includes the Dev container)

- To run the program:
    ```
    cmake -S . -B build -D BUILD_MAIN=ON -D BUILD_TEST_HO=OFF -D BUILD_TEST_RAD=OFF -D BUILD_TEST_BOX=OFF -D BUILD_TEST_STAT=OFF
    cmake build build
    build/main-vmc
    ```
- To run the tests (and save a log)
    ```
    cmake -S . -B build -D BUILD_MAIN=OFF -D BUILD_TEST_XXX=ON -D BUILD_TEST_YYY=OFF
    cmake --build build
    cd build
    make test
    ```
    where `XXX` is one (or more) among `HO`, `RAD`, `BOX`, `STAT`. The tests not set to `ON` should be set to `OFF` (so their keywords should replace `YYY`), so that each variable `BUILD_TEST_KEYWORD` is defined.

# Documentation

Use
```
doxygen
```
then open `html/index.html` in a browser.
There are warnings since some functions (for example, `operator+=` for `Coordinate`) are not documented.
This is by choice, since those are so simple that documenting them would just clutter both the code and the html page.

Do **not** generate a Doxygen configuration file.

# General remarks

1. There is an error message while building the container:
    > Error: there is no registered task type 'cppbuild'. Did you miss installing an extension that provides a corresponding task provider?

    It is completely harmless. See the related [issue](https://github.com/microsoft/vscode-cpptools/issues/6450).
2. The work was split in the following way: Francesco coded the parts `1b`, `1d`, `1f` of the project, Lorenzo coded `1c`, `1e`, `1g`. Each part was first implemented in a branch which was then merged after a pull request. Also there may be some cleanup or bugfixing branches here and there.
