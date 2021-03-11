import numpy as np

import itertools as it
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd


#
# class BayesianAgent:
#     def __init__(self, game_id, money, factor,
#                  a_contrib, b_contrib, a_disclose, b_disclose, lr_disclose, lr_contrib):
#         self.factor = factor
#         self.game_id = game_id
#         self.money = money
#         self.a_contrib = a_contrib
#         self.b_contrib = b_contrib
#         self.a_disclose = a_disclose
#         self.b_disclose = b_disclose
#         self.lr_contrib = lr_contrib
#         self.lr_disclose = lr_disclose
#         self.q = 0
#
#     def contribute(self, opponent_disclosed):
#         return np.round((np.random.beta(a=self.a_contrib, b=self.b_contrib)) * self.money)
#
#     def disclose(self):
#         p_disclose = np.random.beta(a=self.a_disclose, b=self.b_disclose)
#         return np.random.choice([0, 1], p=[1-p_disclose, p_disclose]), p_disclose
#
#     def update_disclosure_posterior(self, opponent_disclosed):
#         self.a_disclose += self.lr_disclose * opponent_disclosed
#         self.b_disclose += self.lr_disclose * (not opponent_disclosed)
#
#     def update_contrib_posterior(self, reward):
#         self.a_contrib += self.lr_contrib * (reward >= self.q)
#         self.b_contrib += self.lr_contrib * (reward <= self.q)
#         self.q += .5 * (reward - self.q)


# class KLevelAgent:
#     def __init__(self, game_id, money, factor,
#                  a_disclose, b_disclose, lr_disclose, temp, k_level):
#         self.factor = factor
#         self.game_id = game_id
#         self.money = money
#         self.init_endowment = money
#         self.a_disclose = a_disclose
#         self.b_disclose = b_disclose
#         self.lr_disclose = lr_disclose
#         self.k_level = k_level
#         self.temp = temp
#
#     def contribute(self):
#         if k_level == 0:
#             contribution = np.random.choice(range(self.money))
#         if k_level == 1:
#             contrib, opponent_contrib, expected = [], [], []
#
#             for c1, c2 in it.product(
#                     range(1, self.money+1), range(1, self.money+1)):
#                 for f in self.possible_factors:
#                     expected.append(
#                         ((self.factor*c1 + f*c2)/2 + self.money - c1) * (1/self.init_endowment)
#                     )
#                     contrib.append(c1)
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
    def __init__(self, money, factor, possible_factors, alpha, lr_contrib, beta):
        self.factor = factor
        self.money = money
        
        self.lr_contrib = lr_contrib
        self.alpha = alpha
        self.beta = beta

        self.q = {k: v for k in [True, False]
                  for v in [
                      {k2: 0 for k2 in [None, ] + possible_factors}
        ]}

        self.y_contrib = {k: v for k in [True, False]
                    for v in [
                        {k2: np.ones((1, money+1))[0] for k2 in [None, ] + possible_factors}
         ]}

    def contribute(self, opp_factor, disclosed):
        p_opp = stats.dirichlet(self.y_contrib[disclosed][opp_factor]).mean()
        expected_rewards = np.zeros((self.money, self.money))

        if opp_factor is None:
            opp_factor = 1

        for idx1, c1 in enumerate(range(1, self.money+1)):
            for idx2, c2 in enumerate(range(1, self.money + 1)):
                expected_rewards[idx1, idx2] = \
                    ((self.factor*c1 + opp_factor*c2)/2 - c1) * p_opp[idx2]

        max_er = np.max(expected_rewards, axis=1)

        return np.random.choice(range(1, self.money+1), p=self.softmax(max_er))

    def disclose(self):
        q = np.array([
            np.mean(list(self.q[False].values())),
            np.mean(list(self.q[True].values()))
        ])

        disclosed = np.random.choice([False, True], p=self.softmax(q))
        return disclosed, self.factor if disclosed else None

    def update_contrib_posterior(self, disclosed, opp_factor, opp_contrib):
        self.y_contrib[disclosed][opp_factor][opp_contrib-1] += self.lr_contrib

    def learn(self, reward, disclosed, opp_type):
        self.q[disclosed][opp_type] += self.alpha * (reward - self.q[disclosed][opp_type])

    def softmax(self, values):
        return np.exp(values*self.beta)/np.sum(np.exp(values*self.beta))


def generate_agents(n_agents, types, money, agent_class):
    agents = []
    factors = [types[0], ] * (n_agents//2) + [types[1], ] * (n_agents//2)
    np.random.shuffle(factors)
    alphas = np.random.beta(1, 1, size=n_agents).tolist()
    betas = np.random.gamma(1.2, 5, size=n_agents).tolist()
    for _ in range(n_agents):
        agents.append(
            agent_class(money=money, factor=factors.pop(), possible_factors=types,
                        alpha=alphas.pop(), lr_contrib=30, beta=betas.pop()
        ))
    return agents


def main():

    # endowment
    money = 10

    # total agent
    n_agents = 100
    n_trials = 500

    types = [.8, 1.2]

    # generate all the agents
    agents = generate_agents(n_agents, types, money, BayesianAgent)

    dd = []

    for t in range(n_trials):

        agent_ids = list(range(n_agents))
        np.random.shuffle(agent_ids)

        for round_id in range(n_agents//2):

            id1 = agent_ids.pop()
            id2 = agent_ids.pop()

            a1 = agents[id1]
            a2 = agents[id2]

            a1_disclosed, f1 = a1.disclose()
            a2_disclosed, f2 = a2.disclose()

            c1 = a1.contribute(opp_factor=f2, disclosed=a1_disclosed)
            c2 = a2.contribute(opp_factor=f1, disclosed=a2_disclosed)

            r1 = ((c1*a1.factor + c2*a2.factor)/2) - c1
            r2 = ((c1*a1.factor + c2*a2.factor)/2) - c2

            a1.update_contrib_posterior(
                opp_contrib=c2, opp_factor=f2, disclosed=a1_disclosed)
            a2.update_contrib_posterior(
                opp_contrib=c1, opp_factor=f1, disclosed=a2_disclosed)

            a1.learn(r1, a1_disclosed, f2)
            a2.learn(r2, a2_disclosed, f1)

            dd.append(
                {'round_id': int(f'{t}{round_id}'), 'id': id1,
                 'c': c1, 't': t, 'f': 'bad' if a1.factor == types[0] else 'good', 'd': a1_disclosed, 'r': r1}
            )
            dd.append(
                {'round_id': int(f'{t}{round_id}'), 'id': id2,
                 'c': c2, 't': t, 'f': 'bad' if a2.factor == types[0] else 'good', 'd': a2_disclosed, 'r': r2}
            )

    df = pd.DataFrame(data=dd)
    sns.set_palette('Set2')

    # prepare data
    # group by bad/good matching
    dff = df[df.groupby('round_id')['f'].transform('nunique') > 1]
    # Plot 1
    ax = plt.subplot(121)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    df_mean = df.groupby(['id', 'f'], as_index=False)['c'].mean()
    df_good = df_mean[df_mean['f'] == 'good']
    df_bad = df_mean[df_mean['f'] == 'bad']

    sem1 = stats.sem(df_bad['c'])
    sem2 = stats.sem(df_good['c'])

    mean1 = np.mean(df_bad['c'])
    mean2 = np.mean(df_good['c'])

    sns.barplot(x=['bad', 'good'], y=[mean1, mean2], ci=None)

    sns.stripplot(x='f', y='c', data=df_mean,
                  linewidth=.6, alpha=.7, edgecolor='w')
    plt.errorbar(
        [0, 1], y=[mean1, mean2], yerr=[sem1, sem2], lw=2.5,
        capsize=3, capthick=2.5, ecolor='black', ls='none', zorder=10)
    plt.title('contribution')
    plt.ylim([0,10])

    # Plot 2
    ax = plt.subplot(122)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    df_mean = df.groupby(['id', 'f'], as_index=False)['d'].mean()
    df_good = df_mean[df_mean['f'] == 'good']
    df_bad = df_mean[df_mean['f'] == 'bad']

    sem1 = stats.sem(df_bad['d'])
    sem2 = stats.sem(df_good['d'])

    mean1 = np.mean(df_bad['d'])
    mean2 = np.mean(df_good['d'])

    sns.barplot(x=['bad', 'good'], y=[mean1, mean2], ci=None)

    sns.stripplot(x='f', y='d', data=df_mean,
                  linewidth=.6, alpha=.7, edgecolor='w')
    plt.errorbar(
        [0, 1], y=[mean1, mean2], yerr=[sem1, sem2], lw=2.5,
        capsize=3, capthick=2.5, ecolor='black', ls='none', zorder=10)
    plt.title('disclosure')
    plt.show()

    # Plot 3
    fig, ax = plt.subplots()

    # sem1 = df[df['f'] == 'bad'].groupby(['t'])['r'].sem()
    # sem2 = df[df['f'] == 'good'].groupby(['t'])['r'].sem()
    #
    # mean1 = df[df['f'] == 'bad'].groupby(['t'])['r'].mean()
    # mean2 = df[df['f'] == 'good'].groupby(['t'])['r'].mean()

    # sns.tsplot(x='t', y='r', hue='f', data=df)
    # plt.show()

    print('exit')

if __name__ == '__main__':
    main()

