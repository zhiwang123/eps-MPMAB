import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot(sizeI, T, num_instances):

    data = pd.read_csv('data/exp2_data_ical=%i' % int(sizeI) + '.%i' % int(T) + 'x%i' % int(num_instances) + '.csv')
    a = data.to_numpy()

    #print(data)
    #print(a)

    M = a[:,0]
    RobustAggMean = a[:,1]
    RobustAggStd = a[:,2]
    IndUCBMean = a[:,3]
    IndUCBStd = a[:,4]

    mpl.style.use('seaborn')
    plt.figure(figsize=(9,8))

    f1 = plt.errorbar(M, RobustAggMean, yerr = RobustAggStd, fmt='-o',
                 capsize=3,
                 elinewidth=2,
                 markeredgewidth=3, label="RobustAgg-Adapted(0.15)")
    f1[-1][0].set_linestyle('--')

    f2 = plt.errorbar(M, IndUCBMean, yerr = IndUCBStd, fmt='-o', color = "peru",
                 capsize=3,
                 elinewidth=2,
                 markeredgewidth=3, label="Ind-UCB")
    f2[-1][0].set_linestyle('--')

    plt.xlabel("M", fontsize=20)
    plt.ylabel("Collective Regret After %i Rounds"%T, fontsize=20)
    plt.xticks(fontsize=19, ticks=[5,10,20])
    plt.yticks(fontsize=19)
    #plt.ylim(-100, 35000)

    plt.legend(loc="upper left", fontsize=19, frameon=True)
    plt.savefig("plots/exp2_ical=%i" % sizeI + "_%i" % T + "x%i" % num_instances + ".png", dpi=300,
                bbox_inches='tight')