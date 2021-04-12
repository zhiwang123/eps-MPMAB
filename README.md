### Multitask Bandit Learning Through Heterogeneous Feedback Aggregation

AISTATS 2021 paper: http://proceedings.mlr.press/v130/wang21e.html

Authors: Zhi Wang*, Chicheng Zhang*, Manish Kumar Singh, 
Laurel D. Riek, and Kamalika Chaudhuri.

Code written by: Manish Kumar Singh and Zhi Wang.

###### Required packages:
- numpy
- matplotlib
- pandas

###### Remarks:
The code can be used to reproduce the two experiments in the paper.
A detailed description of the experimental setup can be found in the appendix
(in this Python implementation, we simplified RobustAgg-Adapted by obviating the 
needs for an initialization phase in which each player pulls each arm once, but one 
can easily add it using the `RobustAgg.InitializationPhase()` function in `module.py`).
Please first create two folders `data/` and `plots/`, 
to which data and plots will be saved, respectively.

###### Usage/Examples:
- To run experiment 1 with `30` generated Bernoulli 0.15-MPMAB problem instances, 
 each of which has `8` subpar arms out of `10` arms and a horizon of `100000` rounds:
`python main.py --exp 1 --time_horizon 100000 --num_subpar_arms 8 --num_instances 30`.

- To run experiment 2 with `30` generated Bernoulli 0.15-MPMAB problem instances
for each value of `M = 5, 10, 20` 
such that each problem instance has `0` subpar arms out of `10` arms and a horizon of `50000` rounds:
`python main.py --exp 2 --time_horizon 50000 --num_subpar_arms 0 --num_instances 30`.




