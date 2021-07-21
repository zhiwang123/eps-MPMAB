import numpy as np

class MAB:
    '''
    Standard single-player MAB instance
    '''
    def __init__(self, arm_count, time_horizon, ground_truth_means):
        self.K = int(arm_count)
        self.T = int(time_horizon)
        self.curr_t = 0
        self.total_reward = 0
        self.ground_truth_means = ground_truth_means
        self.best_arm_mean = np.max(ground_truth_means)
        self.arm_rewards = np.zeros(self.K)
        self.reward_history = []
        self.regret_history = []
        self.empirical_means = np.zeros(self.K)
        self.arm_sample_count = np.zeros(self.K, dtype=np.int32)
        self.upper_confidence_bounds = np.zeros(self.K)
        self.lower_confidence_bounds = np.zeros(self.K)

        self.total_pseudoregret = 0
        self.pseudoregret_history = []

    def ChooseBestArm(self):
        return np.argmax(self.upper_confidence_bounds)

    def UpdateEmpiricalMean(self, a, curr_sample):
        self.curr_t += 1
        self.arm_sample_count[a] += 1
        self.arm_rewards[a] += curr_sample
        self.empirical_means[a] = float(self.arm_rewards[a]) / float(self.arm_sample_count[a])

    def UpdateConfidenceBounds(self, a):
        T = self.T
        self.upper_confidence_bounds[a] = self.empirical_means[a] + np.sqrt(
            float(2 * np.log(T)) / float(self.arm_sample_count[a]))
        self.lower_confidence_bounds[a] = self.empirical_means[a] - np.sqrt(
            float(2 * np.log(T)) / float(self.arm_sample_count[a]))

    def Update(self, a, curr_sample):
        self.UpdateEmpiricalMean(a, curr_sample)
        self.UpdateConfidenceBounds(a)

    def SampleOnce(self):
        for arm in range(self.K):
            curr_sample = np.random.binomial(1, self.ground_truth_means[arm])
            self.Update(arm, curr_sample)
            self.total_reward += curr_sample
            self.reward_history.append(self.total_reward)
            self.regret_history.append(self.best_arm_mean * (len(self.reward_history)) - self.total_reward)
            self.total_pseudoregret += self.best_arm_mean - self.ground_truth_means[arm]
            self.pseudoregret_history.append(self.total_pseudoregret)

    def BanditPlay(self):
        self.SampleOnce()
        for t in range(self.T - self.K):
            curr_arm = self.ChooseBestArm()
            curr_sample = np.random.binomial(1, self.ground_truth_means[curr_arm])
            self.Update(curr_arm, curr_sample)
            self.total_reward += curr_sample
            self.reward_history.append(self.total_reward)
            self.regret_history.append(self.best_arm_mean * (len(self.reward_history)) - self.total_reward)
            self.total_pseudoregret += self.best_arm_mean - self.ground_truth_means[curr_arm]
            self.pseudoregret_history.append(self.total_pseudoregret)

class MPMAB:
    '''
    MPMAB problem instance
    '''
    def __init__(self, a_num_players, a_epsilon, a_arm_count, a_time_horizon, a_sizeI, assumption=True):
        self.m_num_players = int(a_num_players)
        self.m_arm_count = int(a_arm_count)
        self.m_sizeI = int(a_sizeI)
        self.m_epsilon = float(a_epsilon)
        self.m_time_horizon = int(a_time_horizon)
        self.m_ground_truth_means_array = np.zeros((self.m_num_players, self.m_arm_count))

        if assumption == True:

            '''
            When a_epsilon = 0.15 and assumption = True,
            the generated problem instance is a Bernoulli 0.15-MPMAB problem instance
            with the number of subpar arms = a_sizeI.
            See Section 6.2 for more details.
            '''

            b = int(self.m_arm_count - self.m_sizeI)
            self.m_ground_truth_means_array[0, :b] = np.random.uniform(low=0.8, high=0.8 + self.m_epsilon,
                                                                        size=b)
            temp_max = np.max(self.m_ground_truth_means_array[0, :b])
            self.m_ground_truth_means_array[0, b:] = np.random.uniform(low=0, high=temp_max - 5 * self.m_epsilon,
                                                                        size=self.m_sizeI)
            for col in range(self.m_arm_count):
                l = max(0.0, self.m_ground_truth_means_array[0, col] - float(self.m_epsilon) / 2)
                h = min(self.m_ground_truth_means_array[0, col] + float(self.m_epsilon) / 2, 1.0)
                self.m_ground_truth_means_array[1:, col] = np.random.uniform(low=l, high=h, size=self.m_num_players - 1)
        else:
            '''
            An epsilon-MPMAB problem instance
            '''
            seed_ground_truth_means = np.random.uniform(low=0.0, high=1.0, size=self.m_arm_count)
            for col in range(self.m_arm_count):
                l = max(0.0, seed_ground_truth_means[col] - 0.5 * self.m_epsilon)
                h = min(seed_ground_truth_means[col] + 0.5 * self.m_epsilon, 1.0)
                self.m_ground_truth_means_array[:, col] = np.random.uniform(low=l, high=h, size=self.m_num_players)

    def GetMeans(self):
        return self.m_ground_truth_means_array

    def GetNumPlayers(self):
        return self.m_num_players

    def GetHorizon(self):
        return self.m_time_horizon

    def GetArmCount(self):
        return self.m_arm_count

    def GetOptimalReward(self):
        sum = 0
        for i in range(self.m_num_players):
            opt = np.max(self.m_ground_truth_means_array[i])
            sum += opt
        return sum

class RobustAgg:
    def __init__(self, a_epsilon, a_MPMAB):
        self.m_MPMAB = a_MPMAB
        self.m_num_players = self.m_MPMAB.GetNumPlayers()
        self.m_arm_count = self.m_MPMAB.GetArmCount()
        self.m_epsilon = float(a_epsilon)
        self.m_time_horizon = self.m_MPMAB.GetHorizon()
        self.m_ground_truth_means_array = self.m_MPMAB.GetMeans()
        self.m_players = [MAB(self.m_arm_count, self.m_time_horizon, self.m_ground_truth_means_array[r, :]) for r in
                          range(self.m_num_players)]
        self.decisions = np.zeros(self.m_num_players, dtype=np.int64)
        self.player_rewards = np.zeros(self.m_num_players)

    def GetEpsilon(self):
        return self.m_epsilon

    def CalculateUCB_width(self, alpha, n, m, epsilon):
        T = self.m_time_horizon
        ucb_width = np.sqrt(2 * np.log(T) * (float(alpha ** 2) / float(n) + float((1 - alpha) ** 2) / float(m))) + (
                    1 - alpha) * epsilon
        return ucb_width

    '''
    def BestWeightedUCB(self, n, m, X, Y, epsilon):
        T = self.m_time_horizon
        A = 2 * np.log(T) * (float(1) / float(n) + float(1) / float(m))
        B = -4 * np.log(T) / float(m)
        C = 2 * np.log(T) / float(m)
        D = -1 * epsilon
        S = float(4 * A * C * (np.power(D, 2)) - np.power(B * D, 2)) / float(
            4 * np.power(A, 3) - 4 * np.power(A * D, 2))
        if (S < 0):
            if (self.CalculateUCB_width(0, n, m, epsilon) < self.CalculateUCB_width(1, n, m, epsilon)):
                UCB_alpha_star = float(Y) / float(m) + self.CalculateUCB_width(0, n, m, epsilon)
                alpha_star = 0
            else:
                UCB_alpha_star = float(X) / float(n) + self.CalculateUCB_width(1, n, m, epsilon)
                alpha_star = 1
        else:
            if (D >= 0):
                alpha_star = -np.sqrt(S) - float(B) / float(2 * A)
            else:
                alpha_star = np.sqrt(S) - float(B) / float(2 * A)
            if (alpha_star < 0):
                alpha_star = 0
            if (alpha_star > 1):
                alpha_star = 1
            ucb_width = self.CalculateUCB_width(alpha_star, n, m, epsilon)
            UCB_alpha_star = alpha_star * float(X) / float(n) + (1 - alpha_star) * float(Y) / float(m) + ucb_width
        return alpha_star, UCB_alpha_star
    '''

    def BestWeightedUCB(self, n, m, X, Y, epsilon):
        T = self.m_time_horizon
        if epsilon > 0 and n >= (2 * np.log(T) / float(epsilon * epsilon)):
            alpha_star = 1
        else:
            alpha_star = float(n)/float(n+m) * (1 + epsilon * float(m) * np.sqrt(
                1 / (2 * np.log(T)*float(n+m) - float(epsilon * epsilon * n * m))))
        ucb_width = self.CalculateUCB_width(alpha_star, n, m, epsilon)
        return alpha_star, alpha_star * float(X) / float(n) + (1 - alpha_star) * float(Y) / float(m) + ucb_width

    def UpdateConfidenceBounds(self, curr_player):
        for a in range(self.m_arm_count):
            m_other_players = 0
            Y_a = 0
            for p in range(self.m_num_players):
                if (p != curr_player):
                    m_other_players += self.m_players[p].arm_sample_count[a]
                    Y_a += self.m_players[p].arm_rewards[a]
            n = self.m_players[curr_player].arm_sample_count[a]
            X_a = self.m_players[curr_player].arm_rewards[a]
            n = max(n, 1)
            m_other_players = max(m_other_players, 1)
            alpha, ucb = self.BestWeightedUCB(n, m_other_players, X_a, Y_a, self.m_epsilon)
            self.m_players[curr_player].upper_confidence_bounds[a] = ucb

    def IndividualBanditPlay(self):
        for player in self.m_players:
            player.BanditPlay()

    def GetDecisions(self):
        for p in range(self.m_num_players):
            p_arm = self.m_players[p].ChooseBestArm()
            self.decisions[p] = p_arm
        return self.decisions

    def Pull(self):
        for p, p_arm in enumerate(self.decisions):
            p_sample = np.random.binomial(1, self.m_players[p].ground_truth_means[p_arm])
            self.player_rewards[p] = p_sample
        return self.player_rewards

    def ReceiveFeedback(self):
        for p, p_sample in enumerate(self.player_rewards):
            self.m_players[p].Update(self.decisions[p], p_sample)
            self.m_players[p].total_reward += p_sample
            self.m_players[p].reward_history.append(self.m_players[p].total_reward)
            self.m_players[p].regret_history.append(
                self.m_players[p].best_arm_mean * (len(self.m_players[p].reward_history)) - self.m_players[
                    p].total_reward)

            curr_arm = self.decisions[p]
            self.m_players[p].total_pseudoregret += self.m_players[p].best_arm_mean - \
                                                    self.m_players[p].ground_truth_means[curr_arm]
            self.m_players[p].pseudoregret_history.append(self.m_players[p].total_pseudoregret)

        for p in range(self.m_num_players):
            self.UpdateConfidenceBounds(p)

    def CollectiveRegret(self):
        return np.sum(np.array([player.regret_history for player in self.m_players]), axis=0)

    def CollectivePseudoRegret(self):
        return np.sum(np.array([player.pseudoregret_history for player in self.m_players]), axis=0)
