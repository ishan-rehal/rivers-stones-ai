# Stones and Rivers Game

This repository contains the code supporting the code for Agent and Engine for the River and Stones.

## Details
This is a course assignment for the graduate-level Artificial Intelligence course taught by [**Prof. Mausam**](https://www.cse.iitd.ac.in/~mausam/). The assignment documentation can be found on the Course website.

## Rules
You can find the documentation to get all the rules of the game.

## Updates
[15-09-2025] Uploaded the sample files for the C++ users. Please checkout the [Read ME](./c++_sample_files/README.md) for further details. Seperate Submission details will be updated for C++ users.
[19-09-2025] Providing the Self and Opponents time left as an argument to the function from the Game Engine.

## Dependencies
- Python 3.9
- Pygame
- Numpy 
- Scipy


## Setting up the Environment
```sh
conda create --name stones_river python=3.9
conda activate stones_river
pip install -r requirements.txt
```

## Run Instructions
Here are the instructions used to match ai or human players against each other.


## Main Files
- `gameEngine.py`: It is an instance of the game. It can be run locally on your environment. You can run in GUI or CLI mode.
- `agent.py`: It consists of the implementations of the Random Agent. 
- `student_agent.py` : You need to implement your agent here. Some predefined function has been given.

Note: Details for running the C++ agent will be shared later. The same game will be used in the second phase in Assigment 5. And seperate details will be shared for the Assigment 5.

### Human vs Human
```sh
python gameEngine.py --mode hvh
```
### Human vs AI

```sh
python gameEngine.py --mode hvai --circle random
```
### AI vs AI

```sh
python gameEngine.py --mode aivai --circle random --square student
```

### No GUI
```sh
python gameEngine.py --mode aivai --circle random --square student --nogui
```
