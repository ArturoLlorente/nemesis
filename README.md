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
    <a href="https://github.com/ArturoLlorente/nemesis/issues">Report Bug</a>
    ·
    <a href="https://github.com/ArturoLlorente/nemesis/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
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

## Prerequisites

Before you start to work with this project, Docker has to be installed and all dependencies be provided as described in the following sections.

### Install Docker
    
Check the official [Docker](https://docs.docker.com/engine/install/) documentation for installation instructions.


## Installation via Docker Container

### How to build a container

Create a working directory and move all necessary file into that folder. Then run:

```
sudo docker build -f dockerfile -t image-name .
```
### Running a container:

The following command binds ports 8888 and 8008 to the host machine, makes GPUs available and binds the current working directory
to `/app` inside the container:

```
sudo docker run -p 8888:8888 -p 8008:8008 --gpus all --rm -ti --ipc=host -v "$(pwd)":/app container-name
```
### Shelling into a running container

This command opens a shell inside the first docker container that `docker ps` lists:

```
sudo docker exec -it `sudo docker ps | awk 'FNR==2 {print $1}'` /bin/bash
```
### Running a Jupyter Notebook

This command runs a Jupyter Notebook server inside the container:

```
PYTHONPATH=${PYTHONPATH}:/opt/PROPOSAL/build/src/pyPROPOSAL:/usr/lib/nuSQuIDS/resources/python/bindings/ jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/app
```

### Running a Tensorboard

This command runs a Tensorboard server inside the container:

```
tensorboard --port 8008 --logdir=/tmp/tensorboard --bind_all

```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the GNU General Public License v3.0. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Arturo Llorente - [@ArturoLlorente](www.linkedin.com/in/arturo-llorente) - arlloren@ucm.es

Project Link: [https://github.com/ArturoLlorente/nemesis](https://github.com/ArturoLlorente/nemesis)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
