# `qcrypto`: The Quantum Cryptography Simulation Library
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GuillermoFidalgo/QKDP/master.svg)](https://results.pre-commit.ci/latest/github/GuillermoFidalgo/QKDP/master)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

`qcrypto` is a basic, yet flexible quantum cryptography simulation library. It is built upon NumPy, making use of its powerful vectorized operations to allow for the simulation of entangled systems of qubits. Thus, it is built in such a way that it can be easily expanded by the end-user. It offers the following main classes:
* `QstateEnt`: Used for the simulation of a general set of qubits which may or may not be entangled.
* `QstateUnEnt`: Similar to `QstateEnt`, but limited to only simulating unentangled systems of qubits. However, what it lacks in flexibility it makes up in performance: while `QstateEnt` handles $2^N$ coefficients, this class only needs to handle $2\times N$.
* `Agent`: Used to simulate the agents (Alice, Bob, Eve, etc.) in quantum cryptography protocols, allowing for a more intuitive implementation of the desired simulation.

For more information how these features work, you can take a look at [this](https://github.com/GuillermoFidalgo/QKDP/blob/master/notebooks/qCryptoShowcase.ipynb) notebook for a showcase of `QstateEnt` and `QstateUnEnt`, and [this](https://github.com/GuillermoFidalgo/QKDP/blob/master/notebooks/BB84.ipynb) notebook for an example simulation of BB84 using this library.

## Installation
To install `qcrypto` to your Python environment, run the following commands.

```bash
  git clone git@github.com:GuillermoFidalgo/QKDP.git
  cd QKDP
  pip install .
```

If you plan on using `qcrypto` in a Jupyter Notebook, for the last command, run this instead:

```bash
  pip install ".[nb]"
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://guillermofidalgo.github.io"><img src="https://avatars.githubusercontent.com/u/17858942?v=4?s=100" width="100px;" alt="Guillermo A. Fidalgo-Rodríguez"/><br /><sub><b>Guillermo A. Fidalgo-Rodríguez</b></sub></a><br /><a href="#content-GuillermoFidalgo" title="Content">🖋</a> <a href="https://github.com/GuillermoFidalgo/QKDP/commits?author=GuillermoFidalgo" title="Code">💻</a> <a href="#maintenance-GuillermoFidalgo" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/roy-cruz"><img src="https://avatars.githubusercontent.com/u/55238184?v=4?s=100" width="100px;" alt="Roy F. Cruz Candelaria"/><br /><sub><b>Roy F. Cruz Candelaria</b></sub></a><br /><a href="#research-roy-cruz" title="Research">🔬</a> <a href="https://github.com/GuillermoFidalgo/QKDP/commits?author=roy-cruz" title="Code">💻</a> <a href="#content-roy-cruz" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tetianmaz"><img src="https://avatars.githubusercontent.com/u/126983407?v=4?s=100" width="100px;" alt="tetianmaz"/><br /><sub><b>tetianmaz</b></sub></a><br /><a href="#content-tetianmaz" title="Content">🖋</a> <a href="https://github.com/GuillermoFidalgo/QKDP/commits?author=tetianmaz" title="Code">💻</a> <a href="https://github.com/GuillermoFidalgo/QKDP/commits?author=tetianmaz" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
