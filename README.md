This library contains headless versions of a number of commonly used tutoring system environments for training simulated students. 

To create an AI Gym environment for the Fraction Arithmetic tutor use the following commands:

```
import gym
import tutorenvs

env = gym.make('FractionArith-v0')
```
