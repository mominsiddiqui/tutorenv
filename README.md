This library contains headless versions of a number of commonly used tutoring
system environments for training simulated students.  There are currently two
different tutoring systems that can be loaded: the fraction arithmetic tutor
and a multi-column arithmetic tutor. 

The core of each tutor is a "Symbolic" variant of each tutor. This class
maintains the logic for the tutor, including calls for getting the state
description, applying selection, action, inputs (SAIs) which returns
feedback about whether the provided sai is correct/incorrect, and which
provides a method for requesting a demonstration (a valid Sai for the
next step).

The Apprentice Learner Architecture can interface directly with these
symbolic variants of the tutor. 

Next, there are separate classes that wrap these symbolic tutorrs in AI gym
environments that can be loaded by a reinforcement learning algorithm, such as
those in the stable baseline library (e.g., the PPO algorithm). 

Currently, I am exploring multiple representations for the RL tutor:

- Operators model: this is the closest to what the Apprentice Learner would 
  use. In particular, the agent perceives the same hot-one coded state features
  that the AL agent would get. It also has 4 discrete action outputs
  (multi-discrete space), the first is for the selection (all selectable
  interface elements in the tutor), the second discrete output is for an
  operator (copy, add, subtract, multiply, etc., these correspond to prior
  knowledge operators/functions), the next two outputs correspond to the fields
  that get passed as input to the operator (e.g., two other fields in the tutor
  interface that currently have values).

- Digits model, this has the same input as the Operators model (hot-one
  features that AL would get), but has a different action output. Instead, it
  has 4 discrete action outputs (multi-discrete space). The first output is for
  selection (as above), the second is for a digit in the ones place (0-9), the
  third is for a digit in the tens place (0-9), and the fourth is for a digit
  in the hundreds place (0-9).  Depending on the tutor the number of digits
  might be more or less depending on what is necessary to solve the task (e.g.,
  the multi-column arith only has a single digit). 

- Pixel model, This has the same output as the Digits model, but has a different
  input. Instead, the model gets a black and white, pixel representation of the
  tutor interface. It is not identical to the human tutor represention, but it
  is a semi-resonable facimile that includes all the information that exists in
  the human tutor.

These different representations are registered as AI gym models under the names
"FractionArith" and "MultiColumnArith" and version numbers "v0" (for operator
model), "v1" (for digits model), and "v2" for pixel model. As an example of how
to create an operator model for fraction arithmetic and train a PPO model from the
stable baselines package, you would use the following code:
```
import gym
import tutorenvs

env = make_vec_env('MultiColumnArith-v0', n_envs=8)
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000000)
```

See the code in the `sandbox/` folder for examples of how to train different
kinds of agents on these different environments. The
`sandbox/run_al_fractions.py` code shows how to train an Apprentice agent on
the headless fractions tutor.
