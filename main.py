
import mlrose_hiive
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score




def main(seeds):
    queens(seeds)

    kColor(seeds)
    knapsack(seeds)
    NN()

def queens(seeds):

    x = range(2,102,10)
    y=[]
    plt.xlabel('Problem Size')
    plt.ylabel('fitness')
    # code modified from https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
    for i in x:
        prob = mlrose_hiive.generators.QueensGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.RHCRunner(prob, 'fit/prob', seed, np.arange(201),[0])
            ans += a.run()[1]['Fitness'].min()
        ans = -ans/4
        y.append(ans)

    plt.plot(x,y, label='random_hill_climb')
    print('0')

    y1 = []
    for i in x:
        prob = mlrose_hiive.generators.QueensGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.SARunner(prob, 'fit/prob', seed, np.arange(201), [1])
            ans += a.run()[1]['Fitness'].min()
        ans = -ans / 4
        y1.append(ans)

    plt.plot(x, y1, label='simulated_annealing')
    print('1')

    y2 = []
    for i in x:
        prob = mlrose_hiive.generators.QueensGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.GARunner(prob, 'fit/prob', seed, np.arange(201), [100], [.2])
            ans += a.run()[1]['Fitness'].min()
        ans = -ans / 4
        y2.append(ans)

    plt.plot(x, y2, label='genetic_algorithm')
    print('2')

    y3 = []
    for i in x:
        prob = mlrose_hiive.generators.QueensGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.MIMICRunner(prob, 'fit/prob', seed, np.arange(201), [100], [.2], use_fast_mimic=True, max_attempts=5)
            ans += a.run()[1]['Fitness'].min()
        ans = -ans / 4
        y3.append(ans)

    plt.plot(x, y3, label='MIMIC')
    print('3')

    plt.legend()
    plt.show()

    prob = mlrose_hiive.generators.QueensGenerator().generate(0, 100)
    qe = []
    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.RHCRunner(prob, 'fit/iter', seed, np.arange(301),[0]).run()[1]
    qe.append(ans/4)

    ans = None
    for seed in seeds:
        a= mlrose_hiive.runners.SARunner(prob, 'fit/prob', seed, np.arange(301), [1]).run()[1][['Iteration','Fitness','FEvals','Time']]
        print(a)
        if ans is None:
            ans = a
        else:
            ans += a
    qe.append(ans / 4)

    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.GARunner(prob, 'fit/prob', seed, np.arange(301), [100], [.2]).run()[1]
    qe.append(ans / 4)

    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.MIMICRunner(prob, 'fit/prob', seed, np.arange(301), [100], [.2], use_fast_mimic=True, max_attempts=5).run()[1]
    qe.append(ans / 4)
    # end
    plt.xlabel('iterations')
    plt.ylabel('fitness')

    plt.plot(qe[0]['Iteration'],-qe[0]['Fitness'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], -qe[1]['Fitness'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], -qe[2]['Fitness'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], -qe[3]['Fitness'], label='MIMIC')
    plt.legend()
    plt.show()

    plt.xlabel('iterations')
    plt.ylabel('FEvals')

    plt.plot(qe[0]['Iteration'], qe[0]['FEvals'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], qe[1]['FEvals'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], qe[2]['FEvals'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], qe[3]['FEvals'], label='MIMIC')
    plt.legend()
    plt.show()

    plt.xlabel('iterations')
    plt.ylabel('time (sec)')

    plt.plot(qe[0]['Iteration'], qe[0]['Time'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], qe[1]['Time'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], qe[2]['Time'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], qe[3]['Time'], label='MIMIC')
    plt.legend()
    plt.show()

def kColor(seeds):

    x = range(2, 102, 10)
    y = []
    plt.xlabel('Problem Size')
    plt.ylabel('fitness')
    # code modified from https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
    for i in x:
        prob = mlrose_hiive.generators.MaxKColorGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.RHCRunner(prob, 'fit/prob', seed, np.arange(201), [0])
            ans += a.run()[1]['Fitness'].min()
        ans = -ans / 4
        y.append(ans)

    plt.plot(x, y, label='random_hill_climb')
    print('0')

    y1 = []
    for i in x:
        prob = mlrose_hiive.generators.MaxKColorGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.SARunner(prob, 'fit/prob', seed, np.arange(201), [1])
            ans += a.run()[1]['Fitness'].min()
        ans = -ans / 4
        y1.append(ans)

    plt.plot(x, y1, label='simulated_annealing')
    print('1')

    y2 = []
    for i in x:
        prob = mlrose_hiive.generators.MaxKColorGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.GARunner(prob, 'fit/prob', seed, np.arange(201), [100], [.2])
            ans += a.run()[1]['Fitness'].min()
        ans = -ans / 4
        y2.append(ans)

    plt.plot(x, y2, label='genetic_algorithm')
    print('2')

    y3 = []
    for i in x:
        prob = mlrose_hiive.generators.MaxKColorGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.MIMICRunner(prob, 'fit/prob', seed, np.arange(201), [100], [.2],
                                                 use_fast_mimic=True, max_attempts=5)
            ans += a.run()[1]['Fitness'].min()
        ans = -ans / 4
        y3.append(ans)

    plt.plot(x, y3, label='MIMIC')
    print('3')

    plt.legend()
    plt.show()

    prob = mlrose_hiive.generators.MaxKColorGenerator().generate(0, 100)
    qe = []
    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.RHCRunner(prob, 'fit/iter', seed, np.arange(301),[0]).run()[1]
    qe.append(ans/4)

    ans = None
    for seed in seeds:
        a= mlrose_hiive.runners.SARunner(prob, 'fit/prob', seed, np.arange(301), [1]).run()[1][['Iteration','Fitness','FEvals','Time']]
        print(a)
        if ans is None:
            ans = a
        else:
            ans += a
    qe.append(ans / 4)

    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.GARunner(prob, 'fit/prob', seed, np.arange(301), [100], [.2]).run()[1]
    qe.append(ans / 4)

    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.MIMICRunner(prob, 'fit/prob', seed, np.arange(301), [100], [.2], use_fast_mimic=True, max_attempts=500).run()[1]
    qe.append(ans / 4)
    # end
    plt.xlabel('iterations')
    plt.ylabel('fitness')

    plt.plot(qe[0]['Iteration'],-qe[0]['Fitness'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], -qe[1]['Fitness'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], -qe[2]['Fitness'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], -qe[3]['Fitness'], label='MIMIC')
    plt.legend()
    plt.show()

    plt.xlabel('iterations')
    plt.ylabel('FEvals')

    plt.plot(qe[0]['Iteration'], qe[0]['FEvals'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], qe[1]['FEvals'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], qe[2]['FEvals'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], qe[3]['FEvals'], label='MIMIC')
    plt.legend()
    plt.show()

    plt.xlabel('iterations')
    plt.ylabel('time (sec)')

    plt.plot(qe[0]['Iteration'], qe[0]['Time'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], qe[1]['Time'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], qe[2]['Time'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], qe[3]['Time'], label='MIMIC')
    plt.legend()
    plt.show()

def knapsack(seeds):
    x = range(2,202,10)
    y=[]
    plt.xlabel('Problem Size')
    plt.ylabel('fitness')
    # code modified from https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
    for i in x:
        prob = mlrose_hiive.generators.KnapsackGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.RHCRunner(prob, 'fit/prob', seed, np.arange(201),[0])
            ans += a.run()[1]['Fitness'].max()
        ans = ans/4
        y.append(ans)

    plt.plot(x,y, label='random_hill_climb')
    print('0')
    y1 = []
    for i in x:
        prob = mlrose_hiive.generators.KnapsackGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.SARunner(prob, 'fit/prob', seed, np.arange(201), [1])
            ans += a.run()[1]['Fitness'].max()
        ans = ans / 4
        y1.append(ans)

    plt.plot(x, y1, label='simulated_annealing')
    print('1')

    y2 = []
    for i in x:
        prob = mlrose_hiive.generators.KnapsackGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.GARunner(prob, 'fit/prob', seed, np.arange(201), [100], [.2])
            ans += a.run()[1]['Fitness'].max()
        ans = ans / 4
        y2.append(ans)

    plt.plot(x, y2, label='genetic_algorithm')
    print('2')

    y3 = []
    for i in x:
        prob = mlrose_hiive.generators.KnapsackGenerator().generate(0, i)
        ans = 0
        for seed in seeds:
            a = mlrose_hiive.runners.MIMICRunner(prob, 'fit/prob', seed, np.arange(201), [100], [.2], use_fast_mimic=True, max_attempts=10)
            ans += a.run()[1]['Fitness'].max()
        ans = ans / 4
        y3.append(ans)

    plt.plot(x, y3, label='MIMIC')
    print('3')

    plt.legend()
    plt.show()

    prob = mlrose_hiive.generators.KnapsackGenerator().generate(0, 200)
    qe = []
    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.RHCRunner(prob, 'fit/iter', seed, np.arange(301), [0]).run()[1]
    qe.append(ans / 4)

    ans = None
    for seed in seeds:
        a = mlrose_hiive.runners.SARunner(prob, 'fit/prob', seed, np.arange(301), [1]).run()[1][
            ['Iteration', 'Fitness', 'FEvals', 'Time']]
        print(a)
        if ans is None:
            ans = a
        else:
            ans += a
    qe.append(ans / 4)

    ans = 0
    for seed in seeds:
        ans += mlrose_hiive.runners.GARunner(prob, 'fit/prob', seed, np.arange(301), [100], [.2]).run()[1]
    qe.append(ans / 4)

    ans = 0
    for seed in seeds:
        ans += \
        mlrose_hiive.runners.MIMICRunner(prob, 'fit/prob', seed, np.arange(301), [400], [.5], use_fast_mimic=True,
                                         max_attempts=500).run()[1]
    qe.append(ans / 4)
    # end
    plt.xlabel('iterations')
    plt.ylabel('fitness')

    plt.plot(qe[0]['Iteration'], qe[0]['Fitness'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], qe[1]['Fitness'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], qe[2]['Fitness'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], qe[3]['Fitness'], label='MIMIC')
    plt.legend()
    plt.show()

    plt.xlabel('iterations')
    plt.ylabel('FEvals')

    plt.plot(qe[0]['Iteration'], qe[0]['FEvals'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], qe[1]['FEvals'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], qe[2]['FEvals'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], qe[3]['FEvals'], label='MIMIC')
    plt.legend()
    plt.show()

    plt.xlabel('iterations')
    plt.ylabel('time (sec)')

    plt.plot(qe[0]['Iteration'], qe[0]['Time'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], qe[1]['Time'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], qe[2]['Time'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], qe[3]['Time'], label='MIMIC')
    plt.legend()
    plt.show()

def NN():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    # end

    # code from https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
    one_hot = OneHotEncoder()
    y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()
    # end

    # code modified from https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
    qe = []

    ans = mlrose_hiive.runners.NNGSRunner(X_train,y_train,X_test,y_test, '', 0, [200], mlrose_hiive.algorithms.rhc.random_hill_climb,{'activation': [mlrose_hiive.relu], 'scoring':['f1_micro']}, hidden_layer_sizes=[[95,95]], max_attempts=200, activation=mlrose_hiive.relu, learning_rate=[.001], restarts=[0], n_jobs=10, grid_search_scorer_method=f1_score, scorer_method=[f1_score]).run()
    qe.append(ans[1])

    ans = mlrose_hiive.runners.NNGSRunner(X_train, y_train, X_test, y_test, '', 0, [200],
                                          mlrose_hiive.algorithms.sa.simulated_annealing,
                                          {'activation': [mlrose_hiive.relu], 'scoring':['f1_micro']}, hidden_layer_sizes=[[95, 95]],
                                          max_attempts=200, activation=mlrose_hiive.relu,
                                          learning_rate=[.001], n_jobs=10,
                                          grid_search_scorer_method=f1_score, scorer_method=[f1_score]).run()
    qe.append(ans[1])

    ans = mlrose_hiive.runners.NNGSRunner(X_train, y_train, X_test, y_test, '', 0, [200],
                                          mlrose_hiive.algorithms.ga.genetic_alg,
                                          {'activation': [mlrose_hiive.relu], 'population_sizes': [100], 'scoring':['f1_micro']}, hidden_layer_sizes=[[95, 95]],
                                          max_attempts=200, activation=mlrose_hiive.relu,
                                          learning_rate=[.001], n_jobs=20,
                                          grid_search_scorer_method=f1_score, scorer_method=[f1_score], population_sizes=[100]).run()
    qe.append(ans[1])

    ans = mlrose_hiive.runners.NNGSRunner(X_train, y_train, X_test, y_test, '', 0, [200],
                                          mlrose_hiive.algorithms.gd.gradient_descent,
                                          {'activation': [mlrose_hiive.relu], 'population_sizes': [100],
                                           'scoring': ['f1_micro']}, hidden_layer_sizes=[[95, 95]],
                                          max_attempts=200, activation=mlrose_hiive.relu,
                                          learning_rate=[.001], n_jobs=20,
                                          grid_search_scorer_method=f1_score, scorer_method=[f1_score],
                                          population_sizes=[100]).run()
    qe.append(ans[1])
    # end

    plt.xlabel('iterations')
    plt.ylabel('fitness')

    plt.plot(qe[0]['Iteration'], -qe[0]['Fitness'], label='random_hill_climb')
    plt.plot(qe[1]['Iteration'], -qe[1]['Fitness'], label='simulated_annealing')
    plt.plot(qe[2]['Iteration'], -qe[2]['Fitness'], label='genetic_algorithm')
    plt.plot(qe[3]['Iteration'], -qe[3]['Fitness'], label='backprop')
    plt.legend()
    plt.show()

    # code modified from https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
    print(len(X_train))
    x = range(5, len(X_train), len(X_train)//4)
    y = []

    y1 = []
    for i in x:
        ans = mlrose_hiive.runners.NNGSRunner(X_train[:i], y_train[:i], X_test, y_test, '', 0, [200],
                                              mlrose_hiive.algorithms.rhc.random_hill_climb,
                                              {'activation': [mlrose_hiive.relu], 'scoring': ['f1_micro']},
                                              hidden_layer_sizes=[[95, 95]], max_attempts=200,
                                              activation=mlrose_hiive.relu, learning_rate=[.001], restarts=[0],
                                              n_jobs=10, grid_search_scorer_method=f1_score,
                                              scorer_method=[f1_score]).run()
        ans = ans[1]['Fitness'].min()
        ans = -ans
        y1.append(ans)
    y.append(y1)

    y1 = []
    for i in x:
        ans = mlrose_hiive.runners.NNGSRunner(X_train[:i], y_train[:i], X_test, y_test, '', 0, [200],
                                              mlrose_hiive.algorithms.sa.simulated_annealing,
                                              {'activation': [mlrose_hiive.relu], 'scoring': ['f1_micro']},
                                              hidden_layer_sizes=[[95, 95]],
                                              max_attempts=200, activation=mlrose_hiive.relu,
                                              learning_rate=[.001], n_jobs=10,
                                              grid_search_scorer_method=f1_score, scorer_method=[f1_score]).run()
        ans = ans[1]['Fitness'].min()
        ans = -ans
        y1.append(ans)
    y.append(y1)

    y1 = []
    for i in x:
        ans = mlrose_hiive.runners.NNGSRunner(X_train[:i], y_train[:i], X_test, y_test, '', 0, [200],
                                              mlrose_hiive.algorithms.ga.genetic_alg,
                                              {'activation': [mlrose_hiive.relu], 'population_sizes': [100],
                                               'scoring': ['f1_micro']}, hidden_layer_sizes=[[95, 95]],
                                              max_attempts=200, activation=mlrose_hiive.relu,
                                              learning_rate=[.001], n_jobs=20,
                                              grid_search_scorer_method=f1_score, scorer_method=[f1_score],
                                              population_sizes=[100]).run()
        ans = ans[1]['Fitness'].min()
        ans = -ans
        y1.append(ans)
    y.append(y1)

    y1 = []
    for i in x:
        ans = mlrose_hiive.runners.NNGSRunner(X_train[:i], y_train[:i], X_test, y_test, '', 0, [200],
                                              mlrose_hiive.algorithms.gd.gradient_descent,
                                              {'activation': [mlrose_hiive.relu], 'population_sizes': [100],
                                               'scoring': ['f1_micro']}, hidden_layer_sizes=[[95, 95]],
                                              max_attempts=200, activation=mlrose_hiive.relu,
                                              learning_rate=[.001], n_jobs=20,
                                              grid_search_scorer_method=f1_score, scorer_method=[f1_score],
                                              population_sizes=[100]).run()
        ans = ans[1]['Fitness'].min()
        ans = -ans
        y1.append(ans)
    y.append(y1)
    # end

    plt.xlabel('samples')
    plt.ylabel('fitness')

    plt.plot(x, y[0], label='random_hill_climb')
    plt.plot(x, y[1], label='simulated_annealing')
    plt.plot(x, y[2], label='genetic_algorithm')
    plt.plot(x, y[3], label='backprop')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main([34,0,67,84])


