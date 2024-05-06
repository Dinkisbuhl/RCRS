# RCRS-Simulator
 An addition to the repository found at https://github.com/abdalla1912mohamed/Deep_reinforcement_learning_Bachelor_project/tree/main

NOTE: Some of this is not our code, the files we have created ourselves are:
Dyna_Q_NoRandom (but this is just taking some code out of Dyna_Q)
Dyna_Q2
QLearnAdv
DQN2

This project is a pathfinding simulation, where the seeker is not provided a map of the environment, and must instead trial-and-error to avoid walls and find its way to the target goal. There are a few different algorithms, some provided by the original author, some created by us. The goal of the project is to create better alternatives to the algorithms created by the original author, and to show improvement metrics. Dyna_Q is meant to be compared to QLearnAdv, and DQN is meant to be compared to DQN2. This is explained in more detail in our presentation slides.

Setup:
To run this project, you may first have to download some dependencies. There is a setup script inside the "Python Codes" folder, which when run should install all if not nearly everything you need for running. In order, run:
 -python setup.py
 -pip install -r requirements. txt

Additionally, running any of the scripts will tell you what you are missing as an error, but here is a list of possibly conflicting dependencies:
gym==0.9.7
keras==2.14.0
matplotlib==3.8.4
numpy==1.26.4
pygame==2.5.2
pytest==8.2.0
scipy==1.13.0
setuptools==59.6.0
tensorflow==2.14.0

Instructions to run each algorithm:

Open up the "Python Codes" folder in the Terminal. From here, you can run each of the following scripts with python ______.py where the blank space is name of the script. This is a list of the scripts relevant to our project and presentation:
Dyna_Q (Made by original author)
QLearnAdv (Made by us)
DQN (Made by original author)
DQNAdv (Made by us)

Each of these scripts has some alternative options when running. First, there is a line near the top that says: self.env = gym.make('pathfinding-obstacle-25x25-v0'). In DQN2 this is found on line 107. Inside of the quotes, that option can be replaced with any of the following:
pathfinding-obstacle-25x25-v0
pathfinding-obstacle-11x11-v0
pathfinding-free-9x9-v0

These are different environments that the scripts can be run in. It is recommended to not run the 25x25 environment for either DQN script, as those are too big to be completed in any timely fashion. 

Also, to see outputs from these scripts to a file, change the line: with open(r'training_test.txt','w+') as f:. Inside of the quotes, put whatever .txt file you want for the output, though first make said .txt file in the "Python Codes" folder.
