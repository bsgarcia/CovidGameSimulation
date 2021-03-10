import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import scipy.stats as stats

#
# class DeterministicAgent:
#     def __init__(self, game_id, money, factor, possible_factors):
#         self.factor = factor
#         self.possible_factors = possible_factors
#         self.game_id = game_id
#         self.money = money
#         self.a = 10
#         self.b = 10
#
#     def learn(self, opponent_factor):
#         if opponent_factor==.8:
#             self.a += 1
#             return np.random.beta(a=self.a, b=self.b)*10
#         elif opponent_factor is None:
#             self.b += 1
#
#         else:
#
#
#
#     def contribute(self, opponent_disclosed, opponent_factor):
#
#
#
#         else:
#             contrib, opponent_contrib, expected = [], [], []
#
#             for c1, c2 in it.product(
#                     range(0, self.money+1), range(0, self.money+1)):
#                 for f in self.possible_factors:
#                     expected.append(
#                         (self.factor*c1 + f*c2)/2 + self.money - c1
#                     )
#                     contrib.append(c1)
#                     opponent_contrib.append(c2)
#
#             expected = np.array(np.round(expected, 1))
#             contrib = np.array(contrib)
#
#             possible_contrib = contrib[np.max(expected) == expected]
#
#             if len(possible_contrib) > 1:
#                 print(
#                     'Warning: multiple contributions value maximize expected reward'
#                 )
#
#             return np.random.choice(possible_contrib)
#
#     def disclose(self):



class BayesianAgent:
    def __init__(self, game_id, money, factor, param_contrib, alpha, lr_contrib, beta):
        self.factor = factor
        self.game_id = game_id
        self.money = money
        self.param_contrib = param_contrib
        self.lr_contrib = lr_contrib
        self.alpha = alpha
        self.beta = beta

        self.q = {k: v for k in [True, False]
                  for v in [
                      {k2: 0 for k2 in [None, ] + [1.2, .8]}
        ]}

    def contribute(self):
        return np.round((np.random.beta(*self.param_contrib)) * self.money)

    def disclose(self):
        p_disclose = 1/(1+(np.exp(self.beta *
            np.mean(list(self.q[False].values()))
            - np.mean(list(self.q[True].values()))
        )))
        disclosed = np.random.choice([False, True], p=[1-p_disclose, p_disclose])
        return disclosed, self.factor if disclosed else None

    def update_contrib_posterior(self, reward, disclosed, opp_type):
        self.param_contrib[0] += self.lr_contrib * (reward >= self.q[disclosed][opp_type])
        self.param_contrib[1] += self.lr_contrib * (reward <= self.q[disclosed][opp_type])
        self.q[disclosed][opp_type] += self.alpha * (reward - self.q[disclosed][opp_type])


class KLevelAgent:
    def __init__(self, game_id, money, factor,
                 a_disclose, b_disclose, lr_disclose, temp, k_level):
        self.factor = factor
        self.game_id = game_id
        self.money = money
        self.init_endowment = money
        self.a_disclose = a_disclose
        self.b_disclose = b_disclose
        self.lr_disclose = lr_disclose
        self.k_level = k_level
        self.temp = temp

    def contribute(self):
        if k_level == 0:
            contribution = np.random.choice(range(self.money))
        if k_level == 1:
            contrib, opponent_contrib, expected = [], [], []

            for c1, c2 in it.product(
                    range(1, self.money+1), range(1, self.money+1)):
                for f in self.possible_factors:
                    expected.append(
                        ((self.factor*c1 + f*c2)/2 + self.money - c1) * (1/self.init_endowment)
                    )
                    contrib.append(c1)

            expected = np.array(np.round(expected, 1))
            contrib = np.array(contrib)

            possible_contrib = contrib[np.max(expected) == expected]

            if len(possible_contrib) > 1:
                print(
                    'Warning: multiple contributions value maximize expected reward'
                )

            return np.random.choice(possible_contrib)
#


def generate_agents(n_agents, money, agent_class):
    agents = []
    factors = [.8, ] * (n_agents//2) + [1.2, ] * (n_agents//2)
    np.random.shuffle(factors)
    for g in range(n_agents):
        agents.append(
            agent_class(money=money, factor=factors.pop(),
                        game_id=g, alpha=.5, lr_contrib=1, param_contrib=[1, 1],
                        beta=2
        ))
    return agents


def main():

    # np.random.seed(1)

    # endowment
    money = 10

    n_agents = 100

    agents = generate_agents(n_agents, money, BayesianAgent)

    n_trials = 50

    import itertools as it

    for t in range(n_trials):
        agent_ids = list(range(n_agents))
        np.random.shuffle(agent_ids)

        print(t)

        for _ in range(n_agents//2):
            a1 = agents[agent_ids.pop()]
            a2 = agents[agent_ids.pop()]

            a1_disclosed, a1_type = a1.disclose()
            a2_disclosed, a2_type = a2.disclose()

            c1 = a1.contribute()
            c2 = a2.contribute()

            r1 = ((c1*a1.factor + c2*a2.factor)/2) - c1
            r2 = ((c1*a1.factor + c2*a2.factor)/2) - c2

            a1.money += r1
            a2.money += r2

            a1.update_contrib_posterior(r1, a1_disclosed, a2_type)
            a2.update_contrib_posterior(r2, a2_disclosed, a1_type)


    # a = 1
    # b = 1
    # x = np.linspace(0, 1, 100)
    # plt.plot(x, stats.beta.pdf(x, a, b))
    # plt.ylim([-.08, 1.08])
    # plt.title('Disclosure priors')
    # plt.show()
    # lr1 = 5
    # lr2 = 5
    # p_disclose_1 = []
    # p_disclose_2 = []
    # contrib1 = []
# contrib2 = []


    # for t in range(nb_of_trials):
    #
    #     # ask to disclose
    #     a1_disclosed, a1_p = a1.disclose()
    #     a2_disclosed, a2_p = a2.disclose()
    #
    #     # save response
    #     p_disclose_1.append(a1_p)
    #     p_disclose_2.append(a2_p)
    #
    #     # update disclosure expectations
    #     a1.update_disclosure_posterior(a2_disclosed)
    #     a2.update_disclosure_posterior(a1_disclosed)
    #
    #     #contribute
    #     c1 = a1.contribute(a2_disclosed)
    #     c2 = a2.contribute(a1_disclosed)
    #
    #     contrib1.append(c1)
    #     contrib2.append(c2)
    #
    #     #update money pot for each agent
    #     r1 = ((c1*factor1 + c2*factor2)/2) - c1
    #     r2 = ((c1*factor1 + c2*factor2)/2) - c2
    #     a1.money += r1
    #     a2.money += r2
    #     a1.update_contrib_posterior(r1)
    #     a2.update_contrib_posterior(r2)
    #
    #     plt.hist(np.random.beta(a1.a_disclose, a1.b_disclose, size=1000))
    #     plt.title('Agent 1 disclosure posteriors')
    #     plt.hist(np.random.beta(a2.a_disclose, a2.b_disclose, size=1000))
    #     plt.title('Agent 2 disclosure posteriors')
    #     plt.show()

    # plt.plot(range(nb_of_trials), p_disclose_1, label=f'Agent 1 ')
    # plt.plot(range(nb_of_trials), p_disclose_2, label=f'Agent 2')
    # plt.xlabel('Trials')
    # plt.ylabel('P(disclose)')
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()
    #
    # plt.plot(range(nb_of_trials), contrib1, label=f'Agent 1 = {lr1}')
    # plt.plot(range(nb_of_trials), contrib2, label=f'Agent 2 = {lr1}')
    # plt.xlabel('Trials')
    # plt.ylabel('Contribution')
    # plt.legend()
    # plt.ylim([0, 10])
    # plt.show()

if __name__ == '__main__':
    main()

