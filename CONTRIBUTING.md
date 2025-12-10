# Contributing to the Heart Failure Classification project

This file outlines how to propose changes to the Heart Failure Classification project. 

### Fixing typos

Small typos or grammatical errors in documentation may be edited directly using
the GitHub web interface, as long as the changes are made in the _source_ file.

*  Correct: you edit a docstring or comment in a `.py` file below `src/` directory.
*  Incorrect: you edit a generated documentation file, such as a `.html` under `docs/_build/`.

### Prerequisites

Before you make a substantial pull request, you should always file an issue and
make sure someone from the team agrees that it's a problem. If you've found a
bug, create an associated issue and report the issue with enough detail so we can provide help faster.

### Pull request process

* We recommend that you create a Git branch for each pull request (PR).  
* New code should follow PEP8 [style guide](https://www.python.org/dev/peps/pep-0008/).

### Developer notes

#### Developer dependencies

* `conda` (version 23.9.0 or higher)
* `conda-lock` (version 2.5.7 or higher)

#### Adding a new dependency

1. Add the dependency to the `environment.yml` file on a new branch.

2. Run `conda-lock -k explicit --file environment.yml -p linux-64` to update the `conda-linux-64.lock` file.

3. Re-build the Docker image locally to ensure it builds and runs properly.

4. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

5. Update the `docker-compose.yml` file on your branch to use the new container image (make sure to update the tag specifically).

6. Send a pull request to merge the changes into the `main` branch.

### Code of Conduct

Please note that this project is released with a [Contributor Code of
Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to
abide by its terms.

### Attribution

These contributing guidelines were adapted from the [dplyr contributing guidelines](https://github.com/tidyverse/dplyr/blob/master/.github/CONTRIBUTING.md).