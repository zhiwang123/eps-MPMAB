import numpy as np
import module as mx
import pandas as pd
import argparse
import plot
import plot2

# Experiment 1
def compareRegretSizeI(sizeI, T, num_instances):
    K = 10
    M = 20
    ground_truth_epsilon = 0.15

    data = pd.DataFrame()

    for i in range(num_instances):
        problem_instance = mx.MPMAB(a_num_players=M, a_epsilon=ground_truth_epsilon, a_arm_count=K, a_time_horizon=T,
                                 a_sizeI=sizeI, assumption=True)

        # Algorithm 1: robust aggregation
        algRobustAgg = mx.RobustAgg(ground_truth_epsilon, problem_instance)

        # Algorithm 2: baseline
        algBaseline = mx.RobustAgg(ground_truth_epsilon, problem_instance)

        # Algorithm 3: naive aggregation
        algNaiveAgg = mx.RobustAgg(0.0, problem_instance)

        for t in range(T):
            algRobustAgg.GetDecisions()
            algRobustAgg.Pull()
            algRobustAgg.ReceiveFeedback()

            algNaiveAgg.GetDecisions()
            algNaiveAgg.Pull()
            algNaiveAgg.ReceiveFeedback()

        algBaseline.IndividualBanditPlay()

        df = pd.DataFrame({'Round':np.arange(T, dtype=int),
                           'Cumulative Collective Regret':algRobustAgg.CollectiveRegret(),
                           'Cumulative Collective Pseudo-Regret':algRobustAgg.CollectivePseudoRegret(),
                           'Algorithm':'RobustAgg-Adapted', 'instance_num': int(i)})
        df = pd.concat([df, pd.DataFrame(
            {'Round': np.arange(T, dtype=int), 'Cumulative Collective Regret': algBaseline.CollectiveRegret(),
             'Cumulative Collective Pseudo-Regret':algBaseline.CollectivePseudoRegret(),
             'Algorithm': 'Ind-UCB', 'instance_num': int(i)})])
        df = pd.concat([df, pd.DataFrame(
            {'Round': np.arange(T, dtype=int), 'Cumulative Collective Regret': algNaiveAgg.CollectiveRegret(),
             'Cumulative Collective Pseudo-Regret':algNaiveAgg.CollectivePseudoRegret(),
             'Algorithm': 'Naive-Agg', 'instance_num': int(i)})])

        data = pd.concat([data, df])

    data.to_csv('data/exp1_data_ical=%i'%int(sizeI) + '.%i'%int(T) + 'x%i'%int(num_instances) +'.csv',index=False)

# Experiment 2
def compareDependenceOnM(sizeI, T, num_instances):
    K = 10
    M = [5, 10, 20]
    ground_truth_epsilon = 0.15

    robustAggMean = np.zeros((len(M)))
    robustAggStd = np.zeros(len(M))
    indUCBMean = np.zeros((len(M)))
    indUCBStd = np.zeros(len(M))

    for m in M:

        regret_m = np.zeros((2, num_instances))

        for i in range(num_instances):
            problem_instance = mx.MPMAB(a_num_players=m, a_epsilon=ground_truth_epsilon, a_arm_count=K,
                                        a_time_horizon=T,
                                        a_sizeI=sizeI, assumption=True)

            # Algorithm 1: robust aggregation
            algRobustAgg = mx.RobustAgg(ground_truth_epsilon, problem_instance)

            # Algorithm 2: baseline
            algBaseline = mx.RobustAgg(ground_truth_epsilon, problem_instance)

            for t in range(T):
                algRobustAgg.GetDecisions()
                algRobustAgg.Pull()
                algRobustAgg.ReceiveFeedback()
            algBaseline.IndividualBanditPlay()

            regret_m[0][i] = algRobustAgg.CollectivePseudoRegret()[-1]
            regret_m[1][i] = algBaseline.CollectivePseudoRegret()[-1]

        robustAggMean[M.index(m)] = np.mean(regret_m[0])
        robustAggStd[M.index(m)] = np.std(regret_m[0])
        indUCBMean[M.index(m)] = np.mean(regret_m[1])
        indUCBStd[M.index(m)] = np.std(regret_m[1])

    data = pd.DataFrame({'M': M,
                       'RobustAgg Mean': robustAggMean, 'RobustAgg Std': robustAggStd,
                       'ind-UCB Mean': indUCBMean, 'ind-UCB Std': indUCBStd})
    data.to_csv('data/exp2_data_ical=%i' % int(sizeI) + '.%i' % int(T) + 'x%i' % int(num_instances) + '.csv',
                index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-E', '--exp', help='which experiment to run (options: 1 and 2, default: 1)',
                        type=int, choices={1,2}, default=1, required=True)
    parser.add_argument('-T', '--time_horizon', help='the time horizon T (default: 50000)',
                        type=int, default=50000, required=True)
    parser.add_argument('-I', '--num_subpar_arms', help='the number of subpar arms (an integer between 0 and K-1,'
                                                           'default: 0)',
                        type=int, default = 0, required=True)
    parser.add_argument('-R', '--num_instances', help='generate multiple problem instances and compute the average'
                                                         'performance over the instances (default: 10)',
                        type=int, default=10, required=True)
    args = parser.parse_args()

    if args.exp == 1:
        compareRegretSizeI(args.num_subpar_arms, args.time_horizon, args.num_instances)
        plot.plot(args.num_subpar_arms, args.time_horizon, args.num_instances)
    else:
        compareDependenceOnM(args.num_subpar_arms, args.time_horizon, args.num_instances)
        plot2.plot(args.num_subpar_arms, args.time_horizon, args.num_instances)