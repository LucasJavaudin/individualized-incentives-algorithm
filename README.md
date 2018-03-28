Individualized Incentives Algorithm
===================================

Algorithm to compute optimal allocation of individualized incentives in a discrete-choice framework

Structure of the Project
------------------------

The pdfs are stored in the directory _doc_.

The python files are stored in the directory _python_.

How-To Run the Algorithm
------------------------

There is currently no GUI implemented to run the algorithm.
You can run the algorithm by typing the following command line in a terminal.
> python -i /path/to/file/incentives_algorithm.py

Then, you can type commands in the python interpreter.

### Command full_simulation()

This command generate random data, sort the data, remove the Pareto-dominated alternatives and run the algorithm.

#### Output

If no error occurred, 4 text files and 6 graphs are generated and saved in directory _files/_.
If you run the command a second time, the previous files are deleted.
You can store the files in a different directory using the parameter _directory_ (see example below).
- _data.txt_: file with the generated data (utility and energy gains for all alternatives)
- _data_characteristics.txt: file with some characteristics on the data (number of individual, total energy gains possible, etc.)
- _results.txt: file with the results of the algorithm (final choice and amount of incentives for all individuals)
- _results_characteristics.txt_: file with some characteristics on the results (expenses, energy gains, number of iterations, etc.)
- _efficiency_curve.png_: graph plotting total energy gains against expenses
- _efficiency_evolution.png_: graph plotting efficiency of the jumps against iterations
- _incentives_evolution.png_: graph plotting amount of incentives of the jumps against iterations
- _energy_gains.png_: graph plotting energy gains of the jumps against iterations
- _bounds.png: graph with lower and upper bounds for total energy gains
- _individuals_who_moved.png_: graph plotting number of individuals who received incentives against iterations
- _individuals_at_first_best.png_: graph plotting number of individuals at first best alternative (alternative with the most energy gains) against iterations

#### Parameters

You can change the value of the parameters used in the generating process:
- _individuals_ (default: 1000): number of individuals generated
- _mean_nb_alternatives_ (default: 10): average number of alternatives per individual
- _use_poisson_ (default: True): if True, the number of alternatives is drawn from a Poisson distribution, else the number of alternatives is fixed to mean_nb_alternatives
- _use_gumbel_ (default: False): if True, stochastic utility is generated from the Gumbel distribution, else the Logistic distribution is used
- _random_utility_parameter_ (default: 10): parameter of the distribution used to generate stochastic utility (Gumbel or Logistic)
- _utility_mean_ (default: 1): mean parameter for the log-normal distribution used to generate the utility of the alternatives
- _utility_sd_ (default: 1): standard-deviation parameter for the log-normal distribution used to generate the utility of the alternatives
- _alpha_ (default: 1): energy consumption of an alternative is defined by alpha * (U ^ gamma) + beta where U is the utility of the alternative
- _beta_ (default: 0): see alpha
- _gamma_ (default: 1): see alpha

#### Example

The following command run the algorithm with 500 individuals and 20 alternatives per individual on average and store the results in the directory _results/_:
> full_simulation(directory='results', individuals=500, mean_nb_alternatives=20)
