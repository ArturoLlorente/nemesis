# nemesis

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ArturoLlorente/nemesis">
    <img src="images/P-ONE_logo.jpg" alt="Logo">
  </a>

  <h3 align="center">Nemesis</h3>

  <p align="center">
    <br />
    <a href="https://github.com/ArturoLlorente/nemesis"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ArturoLlorente/nemesis">View Demo</a>
    ·
    <a href="https://github.com/ArturoLlorente/nemesis">Report Bug</a>
    ·
    <a href="https://github.com/ArturoLlorente/nemesis">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

The Pacific Ocean Neutrino Experiment (P-ONE) is an initiative towards constructing a multi-cubic-kilometer neutrino telescope in the Pacific Ocean to expand our observable window of the Universe to the highest energies.

Nemesis is a GNN based classification tool for event classification at P-One. The repository has a set of different models to generate and classify cascades, tracks and starting tracks for different detector topologies and energy configuration.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To be able to run the code you need to have a working installation of [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/new-project.html). 
After that you can clone the repository and install the requirements.

### Prerequisites

* PyTorch
* PyTorch Geometric
* PyTorch Lightning

### Installation

1. Clone the repo
   ```sh
   git clone
    ```
2. Install requirements
    ```sh
    pip install -r requirements.txt
    ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Container installation

The repository contains a Dockerfile to build a container with all the requirements. To build the container you can run the following command:

```sh
docker build -t nemesis .
```

To run the container you can use the following command:

```sh
docker run -it --rm -v /path/to/data:/data nemesis
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
