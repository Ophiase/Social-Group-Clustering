# Social-Group-Clustering

Tools to help primary school teacher to manage children based on their psychologic profile using spectral clustering (unsupervised learning).

🚧 Warning  : This is still a proof of work. Research has to be done to prove the beneficial/detrimental effects of those tools applied to real life situations.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- Setup
    - [Installation](#installation)
    - [Run](#run)
- [Future Development](#future-development)

## Introduction

In the following we will consider the metrics bellow :
- Agitation : Too much agitation induce
    - Assumption : Affinity is correlated with agitation

### Inside a class

A teacher may be interested in grouping children in a specific way (for projects or spatial placement in class) to optimize the above metrics.

### On multiple classes

## Features

## Setup

### Installation

- conda install of the dependencies :
```sh
conda env create -f environment.yml
```

- automatic install of dependencies :
```sh
./install_dependencies
```

### Run

```sh
make ? # run the client
make ? # run the demo
make ? # run the tests
```

## Future Development

- Monitor the psychological effects of the developped tools in real life situation using different features for the clustering.
    - If the clusters properly identify psychological groups
        - There may be a risk that reducing the affinity between children might induce social isolation ?
        - In the other way, increasing too much the affinity inside psychological clusters may polarise too much the children ?
    - Otherwise
        - We need to investigate our approach.
        - Is our model more proefficient to optimize the above metrics than a random clustering ?

- Monitor the affinity between clusters