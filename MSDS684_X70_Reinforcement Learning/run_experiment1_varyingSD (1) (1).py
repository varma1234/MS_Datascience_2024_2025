import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import pandas as pd
from bandit import Bandit


def run_experiment(mean, std_devs, N):
    for std in std_devs:
        bandit = Bandit(mean, std)
        count = np.arange(0, N)
        data = np.empty(N)
        mean_values = np.empty(N)
        var_values = np.empty(N)
        svar_values = np.empty(N)

        # Generate data points
        bandit.play()  # Initialize
        for i in range(N):
            data[i] = bandit.play()
            mean_values[i], var_values[i], svar_values[i] = bandit.get_statistics()

        # Plot results
        single_bandit_plots(count, data, mean_values, var_values, svar_values, std)

def single_bandit_plots(count, data, mean, var, svar, std):
    description = {
        "iteration": count,
        "value": data,
        "estimated mean": mean,
        "estimated sample variance": var,
        "estimated population variance": svar,
    }
    pddata = pd.DataFrame(description)

    plt.rcParams["figure.figsize"] = [15, 5]
    sns.set(style="darkgrid")

    # Plot estimated mean
    ax1 = sns.lineplot(x="iteration", y="estimated mean", data=pddata)
    ax1.set_title(f"Estimated Mean (std={std})")
    plt.show()

    # Plot estimated sample variance
    ax2 = sns.lineplot(x="iteration", y="estimated sample variance", data=pddata)
    ax2.set_title(f"Estimated Sample Variance (std={std})")
    plt.show()



# plots using seaborn
def single_bandit_plots(count, data, mean, var, svar):
    description = {
        "iteration": count,
        "value": data,
        "estimated mean": mean,
        "estimated sample variance": var,
        "estimated population variance": svar,
    }
    pddata = pd.DataFrame(description)

    plt.rcParams["figure.figsize"] = [15, 5]

    sns.set(style="darkgrid")
    ax1 = sns.scatterplot(x="iteration", y="value", s=5, data=pddata)
    ax1.set(xscale="linear", yscale="linear")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_title("Data")

    plt.show()

    sns.set(style="darkgrid")
    ax2 = sns.lineplot(x="iteration", y="estimated mean", data=pddata)
    ax2.set(xscale="linear", yscale="linear")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_title("Estimated Mean")

    plt.show()

    sns.set(style="darkgrid")
    ax3 = sns.lineplot(x="iteration", y="estimated sample variance", data=pddata)
    ax3.set(xscale="linear", yscale="linear")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_title("Estimated Sample Variance")

    plt.show()

    sns.set(style="darkgrid")
    ax4 = sns.lineplot(x="iteration", y="estimated population variance", data=pddata)
    ax4.set(xscale="linear", yscale="linear")
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    ax4.set_title("Estimated Population Variance")

    plt.show()

    sns.set(style="darkgrid")
    ax5 = sns.lineplot(x="iteration", y="estimated mean", data=pddata)
    ax5.set(xscale="log", yscale="linear")
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    ax5.set_title("Estimated Mean (log x-scale)")

    plt.show()

    sns.set(style="darkgrid")
    ax6 = sns.lineplot(x="iteration", y="estimated sample variance", data=pddata)
    ax6.set(xscale="log", yscale="linear")
    ax6.set_xlabel("")
    ax6.set_ylabel("")
    ax6.set_title("Estimated Sample Variance (log x-scale)")

    plt.show()

    sns.set(style="darkgrid")
    ax7 = sns.lineplot(x="iteration", y="estimated population variance", data=pddata)
    ax7.set(xscale="log", yscale="linear")
    ax7.set_xlabel("")
    ax7.set_ylabel("")
    ax7.set_title("Estimated Population Variance (log x-scale)")

    plt.show()

# Main function
def main():
    seed = None
    N = 10000
    mean = 0
    std_devs = [1, 10, 20]
    random.seed(seed)
    run_experiment(mean, std_devs, N)

if __name__ == "__main__":
    main()

    
