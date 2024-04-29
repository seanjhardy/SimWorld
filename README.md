A GPU-accelerated simulation environment for training generalist autonomous agents.

Python, pytorch

CURRENT ENVIRONMENTS:
FishTank - A 2d procedurally generated fishtank containing food pellets and a physics simulation driven using inverse kinematics.
Each character in the environment recieve a sequence of observations of generated by casting rays out from the characters head.
This 1D RGB image is flattened into a single array, and appended with the character's current actions, their body position 
as a percentage of the tank size in the x and y dimension, their direction in terms of normalised x and y coordinates,
and their velocity normalised by their maximum acceleration.

This input is then computed by the controller and returns an action space of 3 floating point numbers.
Actions correspond to \[forward_thrust (0 - 1), rotational acceleration (0 full counter-clockwise, 0.5 - still, 1 - full clockwise, (TBD)\]

![](https://github.com/seanjhardy/SimWorld/environments/fishTank/gif)

CONTROLLERS:
The primary controller AgentController, is built using a transformer architecture which predicts the 
future observation state t+1 at time t. The architecture uses a context length of 1024 samples, 12 layers and 12 heads, 
and an embedding size of 764 units. Models can be dynamically saved and loaded from the checkpoints folder, using an existing model config if found.
