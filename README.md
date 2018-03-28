Individualized Incentives Algorithm
===================================

Algorithm to compute optimal allocation of individualized incentives in a discrete-choice framework

How-To Run the Algorithm
------------------------

There is currently no GUI implemented to run the algorithm.
You can run the algorithm by typing the command line:
> python -i /path/to/file/incentives_algorithm.py

Then, you can type commands in the python interpreter.

### Command full_simulation()

This command generate random data, sort them, remove the Pareto-dominated alternatives and run the algorithm.

#### Output

If no error occurred, 4 text files and 6 graphs are generated and saved in directory _files/_.
If you run the command a second time, the previous files are deleted.
You can store the files in a different directory using the parameter _directory_ (see example below).
- data.txt: file with the generated data (utility and energy gains for all alternatives)
- data_characteristics.txt: file with some characteristics on the data (number of individual, total energy gains possible, etc.)
- results.txt: file with the results of the algorithm (final choice and amount of incentives for all individuals)
- results_characteristics.txt: file with some characteristics on the results (expenses, energy gains, number of iterations, etc.)
- efficiency_curve.png: graph plotting total energy gains against expenses
- efficiency_evolution.png: graph plotting efficiency of the jumps against iterations
- incentives_evolution.png: graph plotting amount of incentives of the jumps against iterations
- energy_gains.png: graph plotting energy gains of the jumps against iterations
- bounds.png: graph with lower and upper bounds for total energy gains
- individuals_who_moved.png: graph plotting number of individuals who received incentives against iterations
- individuals_at_first_best.png: graph plotting number of individuals at first best alternative (alternative with the most energy gains) against iterations

#### Parameters

You can change the value of the parameters used in the generating process:
- individuals (default: 1000): number of individuals generated
- mean_nb_alternatives (default: 10): average number of alternatives per individual
- use_poisson (default: True): if True, the number of alternatives is drawn from a Poisson distribution, else the number of alternatives is fixed to mean_nb_alternatives
- use_gumbel (default: False): if True, stochastic utility is generated from the Gumbel distribution, else the Logistic distribution is used
- random_utility_parameter (default: 10): parameter of the distribution used to generate stochastic utility (Gumbel or Logistic)
- utility_mean (default: 1): mean parameter for the log-normal distribution used to generate the utility of the alternatives
- utility_sd (default: 1): standard-deviation parameter for the log-normal distribution used to generate the utility of the alternatives
- alpha (default: 1): energy consumption of an alternative is defined by alpha * (U ^ gamma) + beta where U is the utility of the alternative
- beta (default: 0): see alpha
- gamma (default: 1): see alpha

#### Example

The following command run the algorithm with 500 individuals and 20 alternatives per individual on average and store the results in the directory _results/_:
> full_simulation(directory='results', individuals=500, mean_nb_alternatives=20)
