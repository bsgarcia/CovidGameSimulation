import numpy as np
import itertools as it
import time


class BasicAgent:
    def __init__(self, game_id, money, factor, possible_factors):
        self.factor = factor
        self.possible_factors = possible_factors
        self.game_id = game_id
        self.money = money

    def contribute(self):
        raise NotImplementedError

    def disclose(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError


class DeterministicAgent(BasicAgent):
    def __init__(self, game_id, money, factor, possible_factors):
        super().__init__(game_id, money, factor, possible_factors)

    def contribute(self):
        contrib, opponent_contrib, expected = [], [], []
        for c1, c2 in it.product(
                range(0, self.money+1), range(0, self.money+1)):
            for f in self.possible_factors:
                expected.append(
                    (self.factor*c1 + f*c2)/2 + self.money - c1
                )
                contrib.append(c1)
                opponent_contrib.append(c2)
        expected = np.array(expected)
        contrib = np.array(contrib)
        possible_contrib = contrib[np.max(expected) == expected]

        if len(possible_contrib) > 1:
            print('Warning: multiple contributions value maximize expected reward')

        return np.random.choice(possible_contrib)

    def disclose(self):
        pass


class BayesianAgent(BasicAgent):
    def __init__(self, game_id, money, factor, possible_factors, alpha):
        super().__init__(game_id, money, factor, possible_factors)

        self.contribution_dist = contribution_dist
        self.disclose_params = [1, 1]
        # self.beta = beta

        self.disclose_threshold = .5

    # def contribute(self):


    def disclose(self):
        return np.random.beta(self.disclose_params[0], self.disclose_params[1])

    def learn(self, outcome):
        pass


def main():

    np.random.seed(1)

    # endowment
    money = 10

    nb_of_agents_per_group = 50
    nb_of_trials = 20
    factors = np.concatenate([np.ones(nb_of_agents_per_group) * .8,
                              np.ones(nb_of_agents_per_group) * 1.2]).tolist()
    np.random.shuffle(factors)

    for game_id in range(nb_of_agents_per_group):

        factor1 = factors.pop()
        factor2 = factors.pop()
        print('factors: ', factor1, ', ', factor2)

        a1 = RLAgent(money=money, factor=factor1,
                     possible_factors=np.unique(factors), game_id=game_id, alpha=.5)

        a2 = RLAgent(money=money, factor=factor2,
                     possible_factors=np.unique(factors), game_id=game_id, alpha=.5)

        for t in range(nb_of_trials):
            if a1.disclose():
                a2.set_opp_factor(factor1)
            if a2.disclose():
                a1.set_opp_factor(factor2)

            c1 = a1.contribute()
            c2 = a2.contribute()

            a1.money += (c1 + c2)/2 - c1
            a2.money += (c1 + c2)/2 - c2

            a1.learn(c2)
            a2.learn(c1)
            print(a1.expected_opp_contrib)
            print(a2.expected_opp_contrib)
        time.sleep(20)


if __name__ == '__main__':
    main()

