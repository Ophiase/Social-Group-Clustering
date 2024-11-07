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


In the following, the tool considers several key psychological metrics to facilitate optimal groupings:

- **Agitation**: High levels of agitation can disrupt focus and group cohesion.
    - **Assumption**: Children with similar levels of agitation are more likely to have affinity and collaborate effectively.
- **Mental Health**: Measures related to emotional well-being, resilience, and stress tolerance.
    - **Goal**: Group children with similar mental health profiles or distribute varied profiles based on project needs.
- **Social Engagement**: Levels of participation, willingness to collaborate, and social interactions.
    - **Goal**: Create groups with balanced or complementary social engagement levels to support inclusive activities.
- **Learning Styles**: Preferences for hands-on activities, visual aids, or verbal instructions.
    - **Assumption**: Grouping by similar or complementary learning styles can enhance collaborative learning outcomes.
- **Temperament**: General disposition, such as being calm, reactive, or adaptable.
    - **Goal**: Form groups with a mix of temperaments to promote balanced interactions.

### Inside a class

Teachers can use this tool to form groups or arrange seating to achieve specific goals, such as minimizing disruptions, enhancing collaboration, or supporting specific learning needs based on students' psychological profiles.

### On multiple classes

School administrators may apply these tools to ensure a balanced distribution of psychological profiles across classes, fostering a more harmonious environment and equitable resource allocation.

## Features

Clustering

## Setup

### Installation

- Docker installation
```sh
docker build -t Social-Group-Clustering
```

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
# To launch the docker
docker run -it Social-Group-Clustering
```

```sh
make run # run the client
make demo # run the demo
make test # run the tests
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