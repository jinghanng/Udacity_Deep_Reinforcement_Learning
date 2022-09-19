# Project 3: Collaboration and Cooperation (Multi-Agent Deep Deterministic Policy Gradient)

This project is part of the Udacity Deep Reinforcement Learning nanodegree program. The project aims to implement a Multi-Agent Deep Deterministic Policy Gradient algorithm in an agent to play tennis.

## Project Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) is added up, to get a score for each agent. This yields 2 (potentially different) scores. Then the maximum of these 2 scores is taken.

- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

Step 1: Download the code as a zip file or clone the project with Git to your local machine.

Step 2: Install Miniconda 

In the following links, you find all the information to install Miniconda (recommended)

- Download the installer: https://docs.conda.io/en/latest/miniconda.html

- Installation Guide: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Step 3: Configure the local environment in a Linux OS

The environment descriptor file included in this repository describes all the packages required to set up the environment.
Run the following commands to configure it.

The environment file describes the packages that needs to be installed to set up the environment. To configure it, run the command below in the terminal:

```
$ conda env create -f environment-linux.yml
```

Then activate the environment by running

```
$ conda activate drlnd-p3-collab-compet
```

## Instructions

To start running the training, ensure the environment is activated:

```
$ conda activate drlnd-p3-collab-compet
```

To run the notebook 'Tennis.ipynb', in the terminal, 'cd' to the project directory root where the file is located.

Then, execute the command:

```
$ jupyter notebook
```

This would automatically launch the jupyter notebook file 'Tennis.ipynb' in the browser.

Follow the instructions of the notebook to start the training.