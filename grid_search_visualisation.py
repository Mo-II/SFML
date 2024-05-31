from pandas.plotting import parallel_coordinates
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def plot_search_results():
    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_search_results(file):
    """
    Params: 
        file: Path to the CSV file containing grid search results.
    """
    ## Results from grid search
    results = pd.read_csv(file)
    results = results.dropna()

    ## Extract relevant columns
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']

    ## Getting unique hyperparameters
    params = [col for col in results.columns if col.startswith('param_')]
    unique_params = {}

    for param in params:
        unique_values = results[param].unique()
        if len(unique_values) > 1:  # Include parameter if it has more than one unique value
            unique_params[param.replace('param_', '')] = unique_values
    
    #print(params)
    print(unique_params)

    if not unique_params:
        print("No parameters found. Cannot visualize results.")
        return


    # Extract best parameters
    best_params = results.iloc[results['rank_test_score'].idxmin()][params].to_dict()

    #print("Best parameters:")
    #print(best_params)

    ## Ploting results

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(best_params.keys())
    for p_k, p_v in best_params.items():
        masks.append(list(results[ p_k] == p_v))


    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        find_param = p.replace('param_', '')
        x = np.array(unique_params[find_param])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()


file = "./grid_search/grid_search_results_nn.csv"

plot_search_results(file)