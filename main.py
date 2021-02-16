import numpy as np
import itertools as it
import matplotlib.pyplot as plt


class DeterministicAgent:
    def __init__(self, game_id, money, factor, possible_factors, a, b, learning_rate):
        self.factor = factor
        self.possible_factors = possible_factors
        self.game_id = game_id
        self.money = money
        self.a = a
        self.b = b
        self.learning_rate = learning_rate

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


class BayesianAgent:
    def __init__(self, game_id, money, factor,
                 a_contrib, b_contrib, a_disclose, b_disclose, lr_disclose, lr_contrib):
        self.factor = factor
        self.game_id = game_id
        self.money = money
        self.a_contrib = a_contrib
        self.b_contrib = b_contrib
        self.a_disclose = a_disclose
        self.b_disclose = b_disclose
        self.lr_contrib = lr_contrib
        self.lr_disclose = lr_disclose
        self.last_opponent_contrib = 0

    def contribute(self, opponent_disclosed):
        return (np.random.beta(a=self.a_contrib, b=self.b_contrib) + (.1*opponent_disclosed)) * self.money

    def disclose(self):
        p_disclose = np.random.beta(a=self.a_disclose, b=self.b_disclose)
        return np.random.choice([0, 1], p=[1-p_disclose, p_disclose]), p_disclose

    def update_disclosure_posterior(self, opponent_disclosed):
        self.a_disclose += self.lr_disclose * opponent_disclosed
        self.b_disclose += self.lr_disclose * (not opponent_disclosed)

    def update_contrib_posterior(self, opponent_contribution):
        self.a_contrib += self.lr_contrib * (opponent_contribution >= self.last_opponent_contrib)
        self.b_contrib += self.lr_contrib * (opponent_contribution <= self.last_opponent_contrib)
        self.last_opponent_contrib = opponent_contribution


def main():

    # np.random.seed(1)

    # endowment
    money = 10

    number_of_games = 1

    nb_of_trials = 200
    factors = [.8, 1.2]

    a = 1
    b = 1
    lr1 = 10
    lr2 = 2
    p_disclose_1 = []
    p_disclose_2 = []
    contrib1 = []
    contrib2 = []

    for game_id in range(number_of_games):

        factor1 = factors[0]
        factor2 = factors[0]

        a1 = BayesianAgent(money=money, factor=factor1, game_id=game_id,
                           lr_disclose=lr1, lr_contrib=lr1, a_contrib=a+40, a_disclose=a, b_contrib=b, b_disclose=b)

        a2 = BayesianAgent(money=money, factor=factor2, game_id=game_id,
                           lr_disclose=lr2, lr_contrib=lr2, a_contrib=a, a_disclose=a, b_contrib=b, b_disclose=b)

        for t in range(nb_of_trials):

            # ask to disclose
            a1_disclosed, a1_p = a1.disclose()
            a2_disclosed, a2_p = a2.disclose()

            # save response
            p_disclose_1.append(a1_p)
            p_disclose_2.append(a2_p)

            # update disclosure expectations
            a1.update_disclosure_posterior(a2_disclosed)
            a2.update_disclosure_posterior(a1_disclosed)

            #contribute
            c1 = a1.contribute(a2_disclosed)
            c2 = a2.contribute(a1_disclosed)

            a1.update_contrib_posterior(c2)
            a2.update_contrib_posterior(c1)
            contrib1.append(c1)
            contrib2.append(c2)

            #update money pot for each agent
            a1.money += ((c1*factor1 + c2*factor2)/2) - c1
            a2.money += ((c1*factor1 + c2*factor2)/2) - c2

    plt.plot(range(nb_of_trials), p_disclose_1, label=f'Agent 1 ')
    plt.plot(range(nb_of_trials), p_disclose_2, label=f'Agent 2')
    plt.xlabel('Trials')
    plt.ylabel('P(disclose)')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    plt.plot(range(nb_of_trials), contrib1, label=f'Agent 1 = {lr1}')
    plt.plot(range(nb_of_trials), contrib2, label=f'Agent 2 = {lr1}')
    plt.xlabel('Trials')
    plt.ylabel('Contribution')
    plt.legend()
    plt.ylim([0, 10])
    plt.show()

if __name__ == '__main__':
    main()

