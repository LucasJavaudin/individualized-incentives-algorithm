#############
#  Imports  #
#############


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import progressbar
import os

from scipy.spatial import ConvexHull


###########
#  Class  #
###########


class Data:

    """A Data object stores the information on the
    alternatives of the individuals.
    
    A Data object has a list attribute with I elements where I is the number of
    individuals.
    Each element is a J_i x 2 numpy array where J_i is the number of
    alternatives of individual i.
    The attribute individuals is an indicator of the number of individuals in
    the data.
    The attribute alternatives_per_individual indicates the number of 
    alternatives for all individuals.
    The attribute total_alternatives indicates the total number of alternatives.
    The attribute generated_data is True if the data are generated.
    The attribute is_sorted is True if the data are sorted by utility and then
    by energy consumption.
    The attribute pareto_dominated_removed is True if the Pareto-dominated 
    alternatives are removed from the data.
    The attribute efficiency_dominated_removed is True if the
    efficiency-dominated alterantives are removed from the data.
    """

    def __init__(self):
        """Initialize a Data object.
        
        The value of list is an empty list.
        The value of individuals is 0.
        The value of generated_data is False by default.
        The data is uncleaned and unsorted by default.
        """
        self.list = []
        self.individuals = 0
        self.alternatives_per_individual = []
        self.total_alternatives = 0
        self.generated_data = False
        self.is_sorted = False
        self.pareto_dominated_removed = False
        self.efficiency_dominated_removed = False
        self.read_time = None
        self.generating_time = None
        self.output_data_time = None
        self.output_characteristics_time = None
        self.sorting_time = None
        self.pareto_removing_time = None
        self.efficiency_removing_time = None

    def _append(self, narray):
        """Append the specified numpy array to the Data object.

        Increase the number of individuals by 1 and update the number of
        alternatives.
        The numpy array should have 2 columns.
        """
        assert isinstance(narray, np.ndarray), \
                'Tried to append a non np.array object to a Data object'
        assert narray.shape[1] == 2, \
                'Tried to append a numpy array which does not have 2 columns'
        self.list.append(narray)
        # The number of individuals increases by 1.
        self.individuals += 1
        # Count the number of alternatives in the array.
        n = narray.shape[0]
        # Add the number of alternatives for the individual.
        self.alternatives_per_individual.append(n)
        # Update the total number of alternatives.
        self.total_alternatives += n
        # The data are a priori no longer sorted and cleaned.
        self.is_sorted = False
        self.pareto_dominated_removed = False

    def read(self, filename, delimiter=',', comment='#', verbose=True):
        """Read data from an input file.

        Lines can be commented with the specified character.
        There are twice as many uncommented lines as individuals.
        The odd line contains the utility of the alternatives, separated by the
        specified delimiter.
        The even line contains the energy consumption of the alternatives, separated
        by the specified delimiter.

        :filename: string with the name of the file containing the data
        :delimiter: the character used to separated the utility and the energy
        consumption of the alternatives, default is comma
        :comment: line starting with this string are not read, should be a
        string, default is #
        :verbose: if True, a progress bar and some information are displayed during
        the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        # Try to open file filename and return FileNotFoundError if python is unable
        # to open the file.
        try:
            input_file = open(filename, 'r')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if verbose:
            # Print a progress bar of unknown duration.
            bar, counter = _unknown_custom_bar(
                    'Importing data from file', 
                    'lines imported')
        # The variable odd is True when the line number is odd (utility) and false
        # when the line number is even (energy consumption).
        odd = True
        for line in input_file:
            # The line starting by comment are not read.
            if not line.startswith(comment):
                # For odd lines, the values are stored in the numpy array utility.
                if odd:
                    utility = np.fromstring(line, sep=delimiter)
                    odd = False
                # For even lines, the values are stored in the numpy array energy.
                else:
                    energy = np.fromstring(line, sep=delimiter)
                    # Make sure that utility and energy are of the same size.
                    assert utility.shape == energy.shape, \
                            """In the input file, each individuals should have two lines
                            with the same number of values"""
                    # Append the numpy array (utility, energy) to the data
                    # object.
                    line = np.stack((utility, energy), axis=1)
                    self._append(line)
                    odd = True
                    if verbose:
                        # Update the progress bar and increase the counter by 1.
                        bar.update(counter)
                        counter += 1
        input_file.close()
        if verbose:
            bar.finish()
        # Store the time spent to read data.
        self.read_time = time.time() - init_time

    def generate(self, individuals=1000, mean_nb_alternatives=10, utility_mean=1,
                 utility_sd=1, random_utility_parameter=10, alpha=1, beta=0,
                 gamma=1, use_poisson=True, use_gumbel=False, verbose=True):
        """Generate a random data.

        First, the number of alternatives of each individual is generated from a
        Poisson law or is set deterministically.
        For each alternative, utility and energy consumption are randomly generated.
        Utility is decomposed into a deterministic part and a random part.
        The deterministic part is drawn from a log-normal distribution.
        The random part is drawn from a Gumbel distribution or a Logistic
        distribution.
        Energy consumption is a function of the deterministic utility:
        Energy = alpha * deterministic_utility^gamma + beta.

        :individuals: number of individuals in the generated data, should be an
        integer greater than 2, default is 1000
        :mean_nb_alternatives: the parameter of the Poisson law used to generate the
        number of alternatives is (mean_nb_alternatives - 1), should be strictly
        greater than 1, should be an integer if use_poisson is false, default is 10
        :utility_mean: mean parameter when generating the utility of the
        alternatives, default is 1
        :utility_sd: standard-deviation parameter when generating the utility of the
        alternatives, should be positive, default is 1
        :random_utility_parameter: parameter used when generating the random part of
        the utility, should be positive, default is 10
        :alpha: parameter used to represent a multiplicative relation between
        utility and energy consumption, default is 1
        :beta: parameter used to represent an additive relation between utility and
        energy consumption, default is 0
        :gamma: parameter used to represent an exponential relation between utility
        and energy consumption, default is 1
        :use_poisson: if True the number of alternatives is drawn from a modified
        Poisson distribution with parameter (mean_nb_alternatives - 1), else the
        number of alternatives is equal to mean_nb_alternatives for all
        individuals, default is True
        :use_gumbel: if True the Gumbel distribution is used to generate the random
        utility, else the Logistic distribution is used, default is False
        :verbose: if True, a progress bar and some information are displayed during
        the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert individuals > 1 and isinstance(individuals, (int, np.int64)), \
                'The parameter individuals should be an integer greater than 2'
        assert mean_nb_alternatives > 1, \
                """The parameter mean_nb_alternatives should be stricty greater than
                1"""
        assert isinstance(mean_nb_alternatives, int) or use_poisson == True, \
                """The parameter mean_nb_alternatives should be an integer when
                use_poisson is false"""
        assert utility_sd >= 0, \
                'The parameter utility_sd should be positive'
        assert random_utility_parameter >= 0, \
                'The parameter random_utility_parameter should be positive'
        if verbose:
            # Print a progress bar of duration individuals.
            bar = _known_custom_bar(individuals, 'Generating data')
        # The Data object is generated from a random sample
        self.generate_data = True
        if use_poisson:
            # Generate a list with the number of alternatives of all individuals from a
            # Poisson law.
            nb_alternatives = np.random.poisson(mean_nb_alternatives-1, individuals) + 1
        else:
            # All individuals have the same number of alternatives.
            nb_alternatives = np.repeat(mean_nb_alternatives, individuals)
        for i in range(individuals):
            # Generate deterministic utility from a log-normal distribution with mean
            # utility_mean and standard-deviation utility_sd.
            # The number of values generated is equal to the number of alternatives
            # of the individual i
            deterministic_utility = np.random.lognormal(utility_mean, utility_sd,
                    size=nb_alternatives[i])
            # Generate random utility from a Gumbel distribution is use_gumbel is
            # True or from a Logistic distribution if use_gumbel is False.
            # The parameter of the random term is random_utility_parameter.
            if use_gumbel:
                random_utility = np.random.gumbel(0, random_utility_parameter,
                    size=nb_alternatives[i])
            else:
                random_utility = np.random.logistic(0, random_utility_parameter,
                        size=nb_alternatives[i])
            # Total utility is the sum of the deterministic utility and the random
            # utility.
            utility = deterministic_utility + random_utility
            # The energy consumption is a function of the deterministic utility with
            # parameters alpha, beta and gamma.
            energy = alpha * np.power(deterministic_utility, gamma) + beta
            # Append the numpy array (utility, energy) to the data object.
            individual = np.stack((utility, energy), axis=1)
            self._append(individual)
            if verbose:
                # Update the progress bar.
                bar.update(i)
        if verbose:
            bar.finish()
        # Inform that the data were randomly generated.
        self.generated_data = True
        # Store the time spent to generate data.
        self.generating_time = time.time() - init_time

    def output_data(self, filename, delimiter=',', comment='#', verbose=True):
        """Write the data on a file.

        The output file can be read with the function read.

        :filename: string with the name of the file where the data are written
        :delimiter: the character used to separated the utility and the energy
        consumption of the alternatives, default is comma
        :comment: string used for the comments in the output file, should be a
        string, default is #
        :verbose: if True, a progress bar and some information are displayed during
        the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        # Try to open file filename and return FileNotFoundError if python is unable
        # to open the file.
        try:
            output_file = open(filename, 'wb')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if verbose:
            # Print a progress bar of duration the number of individuals.
            bar = _known_custom_bar(self.individuals, 'Writing data to file')
        # Write the data for each individual separately with the numpy.savetxt
        # command.
        for i in range(self.individuals):
            line = np.transpose(self.list[i])
            np.savetxt(output_file, line, fmt = '%-7.4f', header = 'Individual '
                       + str(i+1), delimiter = ',', comments=comment)
            if verbose:
                # Update the progress bar.
                bar.update(i)
        if verbose:
            bar.finish()
        # Store the time spent to output data.
        self.output_data_time = time.time() - init_time

    def output_characteristics(self, filename, verbose=True):
        """Write a file with some characteristics on the data.

        :filename: string with the name of the file where the characteristics 
        are written
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        # Try to open file filename and return FileNotFoundError if python is unable
        # to open the file.
        try:
            output_file = open(filename, 'w')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if verbose:
            print('Writing some characteristics about the data on a file...')
        size=40
        # Display the number of individuals, the total number of alternatives,
        # the average number of alternatives, the minimum number of alternatives
        # and the maximum number of alternatives.
        output_file.write('Individuals and Alternatives'.center(size-1, '='))
        output_file.write(
                          '\nNumber of individuals:'.ljust(size) 
                          + str(self.individuals)
                         )
        output_file.write(
                          '\nTotal number of alternatives:'.ljust(size) 
                          + str(self.total_alternatives)
                         )
        output_file.write(
                          '\nAverage number of alternatives:'.ljust(size) 
                          + str(self.total_alternatives / self.individuals)
                         )
        min_alternatives = min(self.alternatives_per_individual)
        output_file.write(
                          '\nMinimum number of alternatives:'.ljust(size) 
                          + str(min_alternatives)
                         )
        max_alternatives = max(self.alternatives_per_individual)
        output_file.write(
                          '\nMaximum number of alternatives:'.ljust(size) 
                          + str(max_alternatives)
                         )
        output_file.write('\n\n')
        output_file.write('Energy Consumption'.center(size-1, '='))
        # State when all the individuals choose their first alternative.
        init_state = np.zeros(self.individuals, dtype=int)
        # Show total energy consumption at the initial state.
        init_energy = self._total_energy(init_state)
        output_file.write(
                          '\nInitial energy consumption:'.ljust(size) 
                          + str(init_energy)
                         )
        # State when all the individuals choose their last alternative.
        last_state = np.array(self.alternatives_per_individual) - 1
        # Show total energy consumption at the last state.
        last_energy = self._total_energy(last_state)
        output_file.write(
                          '\nMinimum potential energy consumption:'.ljust(size)
                          + str(last_energy)
                         )
        # Show the maximum energy consumption gains that could be achieved.
        max_total_energy_gains = init_energy - last_energy
        output_file.write(
                          '\nPotential energy consumption gains:'.ljust(size) 
                          + str(max_total_energy_gains)
                         )
        output_file.write('\n\n')
        output_file.write('Utility and Surplus'.center(size-1, '='))
        # Show total utility at the initial state.
        init_utility = self._total_utility(init_state)
        output_file.write(
                          '\nInitial consumer surplus:'.ljust(size) 
                          + str(init_utility)
                         )
        # Show total utility at the last state.
        last_utility = self._total_utility(last_state)
        output_file.write(
                          '\nMinimum consumer surplus (gross):'.ljust(size) 
                          + str(last_utility)
                         )
        # Show the maximum budget that is necessary.
        max_expenses = init_utility - last_utility
        output_file.write(
                          '\nMaximum budget necessary:'.ljust(size) 
                          + str(max_expenses)
                         )
        output_file.write('\n\n')
        # Indicate if the data were randomly generated or imported.
        output_file.write('Technical Information'.center(size-1, '='))
        if self.generated_data:
            output_file.write('\nThe data were randomly generated')
        else:
            output_file.write('\nThe data were imported from a file')
        # Indicate if the data are sorted.
        if self.is_sorted:
            output_file.write('\nThe data are sorted')
        else:
            output_file.write('\nThe data are not sorted')
        # Indicate if the Pareto-dominated alternatives are removed.
        if self.pareto_dominated_removed:
            output_file.write('\n' 
                              + str(self.nb_pareto_removed) 
                              + ' Pareto-dominated alternatives were removed')
        else:
            output_file.write('\nThe Pareto-dominated alternatives are not'
                              + ' removed')
        # Store the time spent to output characteristics.
        self.output_characteristics_time = time.time() - init_time

    def remove_pareto_dominated(self, verbose=True):
        """Remove the Pareto-dominated alternatives.

        :verbose: if True, a progress bar and some information are displayed during
        the process, default is True
        
        """
        # Ensure the data are sorted before cleaning.
        if not self.is_sorted:
            self.sort(verbose=verbose)
        # Store the starting time.
        init_time = time.time()
        if verbose:
            bar = _known_custom_bar(self.individuals, 
                                    'Cleaning data (Pareto)')
        # Variable used to count the number of removed alternatives.
        nb_removed = 0
        # For each individual remove the Pareto-dominated alternatives.
        for indiv, line in enumerate(self.list):
            # The first alternative is never Pareto-dominated.
            sorted_line = np.array([line[0]])
            # Add the other alternatives if they are not Pareto-dominated.
            for j in range(len(line)-1):
                j += 1
                # The energy consumption of the alternative should be strictly
                # lower than the energy consumption of the previous non
                # Pareto-dominated alternative.
                if line[j, 1] < sorted_line[-1, 1]:
                    sorted_line = np.append(sorted_line, [line[j]], axis=0)
            # Update the array for the individual.
            self.list[indiv] = sorted_line
            # Count the number of alternatives for the individual.
            n = sorted_line.shape[0]
            # Count the number of removed alternatives.
            nb_removed += self.alternatives_per_individual[indiv] - n
            # Update the number of alternatives of the individual.
            self.alternatives_per_individual[indiv] = n
            if verbose:
                bar.update(indiv)
        # Update the total number of alternatives.
        self.total_alternatives -= nb_removed
        # Store the number of removed alternatives.
        self.nb_pareto_removed = nb_removed
        if verbose:
            bar.finish()
            print('Successfully removed ' 
                  + str(self.nb_pareto_removed) 
                  + ' Pareto-dominated alternatives.'
                 )
        # The Pareto-dominated alternatives are now removed.
        self.pareto_dominated_removed = True
        # Store the time spent to remove the Pareto-dominated alternatives.
        self.pareto_removing_time = time.time() - init_time

    def remove_efficiency_dominated(self, verbose=True):
        """Remove the efficiency-dominated alternatives.

        :verbose: if True, a progress bar and some information are displayed during
        the process, default is True

        """
        # Ensure the data are sorted before cleaning.
        #if not self.is_sorted:
            #self.sort(verbose=verbose)
        # Store the starting time.
        init_time = time.time()
        if verbose:
            bar = _known_custom_bar(self.individuals, 
                                    'Cleaning data (effic.)')
        # Variable used to count the number of removed alternatives.
        nb_removed = 0
        for i in range(self.individuals):
            if verbose:
                bar.update(i)
            # Store the alternatives of the individual.
            alternatives_list = self.list[i]
            # Compute the alternative with the largest utility and with the
            # lowest energy consumption.
            first_choice = np.argmax(alternatives_list[:, 0])
            last_choice = np.argmin(alternatives_list[:, 1])
            # If the choice with the individual best choice is also the social
            # best choice, the individual has only one relevant alternative.
            if first_choice == last_choice:
                self.list[i] = np.array([alternatives_list[first_choice]])
                nb_removed += self.alternatives_per_individual[i] - 1
                self.alternatives_per_individual[i] = 1
            else:
                # If there are only two alternatives, both are non-efficiency
                # dominated.
                if self.alternatives_per_individual[i] == 2:
                    pass
                else:
                    # Compute the convex hull.
                    hull = ConvexHull(alternatives_list)
                    # The vertices are the indices of the points on the convex hull.
                    v = hull.vertices
                    # Compute the position in v of the first and last choice (they
                    # are always on the convex hull).
                    pos_first = [i for i, x in enumerate(v) if x==first_choice][0]
                    pos_last = [i for i, x in enumerate(v) if x==last_choice][0]
                    # The points are in counterclockwise order.
                    if pos_first < pos_last:
                        # The non-efficiency dominated choices are all points of the
                        # convex hull before pos_first and after pos_last.
                        x = v[:pos_first+1]
                        x = np.append(x, v[pos_last:])
                    else:
                        # The non-efficiency dominated choices are all points of the
                        # convex hull between pos_last and pos_first.
                        x = v[pos_last:pos_first+1]
                    if len(x)==0:
                        print(self.list[i])
                        print(pos_first)
                        print(pos_last)
                        print(v)
                        print('\n')
                    # Update the array for the individual.
                    self.list[i] = alternatives_list[x]
                    # Count the number of alternatives for the individual.
                    n = self.list[i].shape[0]
                    # Count the number of removed alternatives.
                    nb_removed += self.alternatives_per_individual[i] - n
                    # Update the number of alternatives of the individual.
                    self.alternatives_per_individual[i] = n
        # Update the total number of alternatives.
        self.total_alternatives -= nb_removed
        # Store the number of removed alternatives.
        self.nb_efficiency_removed = nb_removed
        if verbose:
            bar.finish()
            print('Successfully removed ' 
                  + str(self.nb_efficiency_removed) 
                  + ' efficiency-dominated alternatives.'
                 )
        # The efficiency-dominated alternatives are now removed.
        self.efficiency_dominated_removed = True
        # The Pareto-dominate alternatives are also removed, by definition.
        self.pareto_dominated_removed = True
        # Store the time spent to remove the Pareto-dominated alternatives.
        self.efficiency_removing_time = time.time() - init_time

    def sort(self, verbose=True):
        """Sort the data by utility, then by energy consumption.

        :verbose: if True, a progress bar and some information are displayed during
        the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        if verbose:
            bar = _known_custom_bar(self.individuals,
                                    'Sorting data')
        for indiv, line in enumerate(self.list):
            # The numpy array for each individual is sorted by utility and then
            # by energy consumption.
            sorting_indices = np.lexsort((line[:, 1], -line[:, 0]))
            sorted_line = np.array([line[i] for i in sorting_indices])
            self.list[indiv] = sorted_line
            if verbose:
                bar.update(indiv)
        if verbose:
            bar.finish()
        # The data are now sorted.
        self.is_sorted = True
        # Store the time spent to sort data.
        self.sorting_time = time.time() - init_time

    def _total_utility(self, state):
        """Compute the sum of utilities of a given state.

        :state: a list with all the alternatives of the individuals
        :returns: float indicating the sum of utilities

        """
        total = np.sum(np.array(
                    [array[state[i], 0] for i, array in enumerate(self.list)]
                ))
        return total

    def _total_energy(self, state):
        """Compute the total energy consumption of a given state.

        :state: a list with all the alternatives of the individuals
        :returns: float indicating the total energy consumption

        """
        total = sum([array[state[i], 1] for i, array in enumerate(self.list)])
        return total

    def _get_nb_alternatives(self, individual):
        """Return the number of alternatives of the specified individual.

        :individual: index of the individual considered, should be an integer
        :returns: integer indicating the number of alternatives

        """
        J = self.alternatives_per_individual[individual]
        return J

    def _get_utility(self, individual):
        """Return the array of the utility of the alternatives of the specified
        individual.

        :individual: index of the individual considered, should be an integer
        :returns: array with the utility of the alternatives

        """
        array = self.list[individual][:,0]
        return array

    def _get_energy(self, individual):
        """Return the array of the energy consumption of the alternatives of the 
        specified individual.

        :individual: index of the individual considered, should be an integer
        :returns: array with the energy consumption of the alternatives

        """
        array = self.list[individual][:,1]
        return array

    def _single_jump_efficiency(self, individual, previous_alternative,
            next_alternative):
        """Compute the efficiency of the jump from one alternative to another
        alternative for a specific individual.

        :individual: the index of the considered individual, should be an
        integer
        :previous_alternative: the index of the previous alternative of the 
        individual, should be an integer
        :next_alternative: the index of the previous alternative of the 
        individual, should be an integer
        :returns: a float indicating the efficiency of the jump

        """
        # If the previous alternative is greater than the next alternative (in 
        # terms of individual utility) then the efficiency is 0.
        if previous_alternative >= next_alternative:
            efficiency = 0
        else:
            # Take the utility and the energy consumption of the alternatives of the
            # individual.
            utility = self._get_utility(individual)
            energy = self._get_energy(individual)
            # Save the value of utility and energy consumption for the
            # alternative of the individual.
            previous_utility = utility[previous_alternative]
            previous_energy = energy[previous_alternative]
            # Compute the efficiency.
            efficiency = (previous_energy - energy[next_alternative]) \
                         / (previous_utility - utility[next_alternative])
        return efficiency

    def _individual_jump_efficiencies(self, individual, choice):
        """Compute the efficiency of all the possible jumps of a single individual.

        The efficiency of a jump is defined as the difference in energy consumption
        between two alternatives divided by the difference in utility between these
        two alternatives.
        The possible jumps are all the jumps going from the choice of the individual
        to any other alternatives with a lower individual utility.
        An array is returned with the same length as the number of alternatives of
        the considered individual.
        By convention, the efficiency is 0 if the jump is not possible.

        :individual: the index of the considered individual, should be an 
        integer
        :choice: the index of the choice of the individual, should be an
        integer
        :returns: a numpy array with the efficiencies for all the possible jumps

        """
        # Take the utility and the energy consumption of the alternatives of the
        # individual.
        utility = self._get_utility(individual)
        energy = self._get_energy(individual)
        # Take the number of alternatives of the individual.
        J = self._get_nb_alternatives(individual)
        # Save the value of utility and energy consumption for the choice of the
        # individual.
        current_utility = utility[choice]
        current_energy = energy[choice]
        # If the jump is not possible, the efficiency is 0.
        # If the jump is possible, compute the efficiency.
        efficiencies = [
                        0 if j <= choice 
                        else
                        (current_energy - energy[j]) / (current_utility - utility[j])
                        for j in range(J)
                       ]
        efficiencies = np.array(efficiencies)
        return efficiencies

    def _individual_best_jump(self, individual, choice):
        """Compute the best jump of the individual.

        :individual: the index of the considered individual, should be an integer
        :choice: the index of the choice of the individual, should be an
        integer
        :returns: an integer indicating the resulting alternative of the best 
        jump and a float indicating the efficiency of the best jump

        """
        # Compute the efficiency of all the jumps and take the maximum.
        efficiencies = self._individual_jump_efficiencies(individual, choice)
        max_efficiency = np.max(efficiencies)
        # If the value of the maximum efficiency is 0, that means that no jump is
        # possible so the individual is already at its best alternative.
        if max_efficiency == 0:
            best_alternative = 0
        # Else, the best alternative is the resulting alternative of the jump 
        # with the highest efficiency.
        # In case of ties, python automatically chooses the jump with the lower
        # incentive.
        else:
            best_alternative = np.argmax(efficiencies)
        return max_efficiency, best_alternative

    def _all_best_jump(self, state):
        """Compute the best jump of all individuals and return an array with the
        resulting alternatives and the efficiencies.

        :state: a list or a numpy array with the current choice of all the
        individuals
        :returns: a numpy array with one column per individual and two rows
        (efficiency and resulting alternative of the jumps)

        """
        efficiencies = []
        alternatives = []
        # For each individual, compute the resulting alternative and the 
        # efficiency of the best jump and append the results to the lists.
        for i in range(self.individuals):
            efficiency, alternative = self._individual_best_jump(i, state[i])
            efficiencies.append(efficiency)
            alternatives.append(alternative)
        efficiencies = np.array(efficiencies)
        return efficiencies, alternatives

    def _next_best_jumps(self, state):
        """Compute the efficiency of the best jump of all individuals when the
        efficiency-dominated alternatives are removed .

        :state: a list or a numpy array with the current choice of all the
        individuals
        :returns: a numpy array with the efficiency of the best jumps

        """
        efficiencies = []
        for i in range(self.individuals):
            efficiency = self._next_efficiency(i, state[i])
            efficiencies.append(efficiency)
        efficiencies = np.array(efficiencies)
        return efficiencies

    def _next_efficiency(self, individual, choice):
        """Compute the efficiency of the best jump on one individual when the
        efficiency-dominated alternatives are removed.

        :individual: the index of the considered individual, should be an integer
        :choice: the index of the choice of the individual, should be an
        integer
        :returns: a float indicating the efficiency of the best jump

        """
        # Only try to compute the efficiency if the individuals is not at
        # his last alternative.
        if choice != self.alternatives_per_individual[individual]-1:
            # The best jump is simply the jump to the next alternative.
            max_efficiency = self._single_jump_efficiency(individual, 
                                                          choice, 
                                                          choice+1)
        else:
            max_efficiency = 0
        return max_efficiency

    def _incentives_amount(self, individual, previous_alternative,
            next_alternative):
        """Compute the amount of incentives needed to induce the individual to
        change his choice.

        :individual: the index of the considered individual, should be an 
        integer
        :previous_alternative: the index of the alternative currently chosen by 
        the individual, should be an integer
        :next_alternative: the index of the targeted alternative, should be an 
        integer
        :returns: a float indicating the amount of incentives of the jump

        """
        # Get the utility of the individual at his previous alternative and his 
        # next alternative.
        utilities = self._get_utility(individual)
        previous_utility = utilities[previous_alternative]
        next_utility = utilities[next_alternative]
        # The amount of incentives is defined by the loss in utility.
        incentives = previous_utility - next_utility
        return incentives

    def _energy_gains_amount(self, individual, previous_alternative,
            next_alternative):
        """Compute the amount of energy gains when the specified individual
        change his choice.

        :individual: the index of the considered individual, should be an
        integer
        :previous_alternative: the index of the alternative currently chosen by 
        the individual, should be an integer
        :next_alternative: the index of the targeted alternative, should be an 
        integer
        :returns: a float indicating the amount of energy gains of the jump

        """
        # Get the enery consumption of the individual at his previous
        # alternative and his next alternative.
        energy = self._get_energy(individual)
        previous_energy = energy[previous_alternative]
        next_energy = energy[next_alternative]
        # Compute the amount of energy gains.
        energy_gains = previous_energy - next_energy
        return energy_gains
    
    def run_algorithm(self, budget=np.infty, force=True, verbose=True):
        """Compute the optimal state for a given budget by running the algorithm.

        If the available budget is enough to reach the state where all 
        individuals are at their last alternative and if force is False, then 
        the algorithm is not run and the state is directly returned.

        :budget: should be an integer or a float with the maximum amount of
        incentives to give, default is np.infty (the budget is unlimited).
        :force: if True, force the algorithm to run even if it is not necessary,
        default is True.
        :verbose: if True, a progress bar and some information are displayed 
        during the process, default is True
        :returns: an AlgorithmResults object

        """
        # Running the algorithm only work if the Pareto-dominated alternatives
        # are removed.
        if not self.pareto_dominated_removed:
            self.remove_pareto_dominated(verbose=verbose)
        # The data must be sorted.
        if not self.is_sorted:
            self.sort(verbose=verbose)
        # Store the starting time.
        init_time = time.time()
        if verbose:
            if budget == np.infty:
                bar = _known_custom_bar(
                    self.total_alternatives - self.individuals,
                    'Running algorithm'
                )
            else:
                bar = _known_custom_bar(budget, 
                                        'Running algorithm')
        # Create an AlgorithmResults object where the variables are stored.
        results = AlgorithmResults(self, budget)
        # Compute the amount of expenses needed to reach the state where all
        # individuals are at their last alternative.
        # Return the last state if the budget is enough to reach it.
        if (not force) and (budget >= results.max_expenses):
            results.optimal_state = results.last_state
            results.expenses = results.max_expenses
        else:
            # Compute the efficiency and the resulting alternative of the best 
            # jump of all individuals.
            if self.efficiency_dominated_removed:
                best_efficiencies = self._next_best_jumps(results.optimal_state)
            else:
                best_efficiencies, best_alternatives = \
                    self._all_best_jump(results.optimal_state)
            # Main loop of the algorithm.
            # The loop runs until the budget is depleted or until all the jumps
            # have been done.
            while results.expenses < budget and \
                    sum(best_efficiencies!=0) != 0:
                if verbose:
                    if budget == np.infty:
                        bar.update(np.sum(results.optimal_state))
                    else:
                        bar.update(results.expenses)
                # Increase the number of iterations by 1.
                results.iteration += 1
                # Select the individual with the most efficient jump.
                selected_individual = np.argmax(best_efficiencies)
                # Store information on the jump (selected individual, previous
                # alternative, next alternative).
                previous_alternative = \
                    results.optimal_state[selected_individual]
                if self.efficiency_dominated_removed:
                    next_alternative = previous_alternative + 1
                else:
                    next_alternative = best_alternatives[selected_individual]
                jump_information = [selected_individual,
                                     previous_alternative,
                                     next_alternative]
                results.jumps_history.append(jump_information)
                # Store the efficiency of the jump.
                jump_efficiency = best_efficiencies[selected_individual]
                results.efficiencies_history.append(jump_efficiency)
                # Change the current state according to the new choice of the
                # selected individual.
                results.optimal_state[selected_individual] = next_alternative
                # Update the arrays of the best jumps for the selected
                # individual.
                if self.efficiency_dominated_removed:
                    new_best_efficiency = self._next_efficiency(
                                                        selected_individual, 
                                                        next_alternative
                                                        )
                    best_efficiencies[selected_individual] = new_best_efficiency
                else:
                    new_best_efficiency, new_best_alternative = \
                        self._individual_best_jump(selected_individual,
                                                   next_alternative)
                    best_efficiencies[selected_individual] = new_best_efficiency
                    best_alternatives[selected_individual] = new_best_alternative
                # Increase the expenses by the amount of incentives of the jump.
                incentives = self._incentives_amount(*jump_information)
                results.expenses += incentives
                # Increase the total energy gains by the amount of energy gains
                # of the jump.
                energy_gains = incentives * jump_efficiency
                results.total_energy_gains += energy_gains
                # Store the incentives and the energy gains of the jump.
                results.incentives_history.append(incentives)
                results.energy_gains_history.append(energy_gains)
            # In case of overshot (the expenses are greater than the budget), go
            # back to the previous iteration.
            if results.expenses > budget:
                # Reduce the number of iterations by 1.
                results.iteration -= 1
                # Restore the previous state.
                results.optimal_state[selected_individual] = \
                    previous_alternative
                # Reduce the expenses by the amount of incentives of the last
                # jump.
                results.expenses -= incentives
                # Reduce the total energy gains by the amount of energy gains of
                # the last jump.
                results.total_energy_gains -= energy_gains
                # Remove the last jump of the list and store it in a special
                # variable.
                results.overshot_jump = results.jumps_history[-1]
                results.jumps_history = results.jumps_history[:-1]
                # Remove the last value of incentives, energy gains and
                # efficiency.
                results.efficiencies_history = results.efficiencies_history[:-1]
                results.incentives_history = results.incentives_history[:-1]
                results.energy_gains_history = results.energy_gains_history[:-1]
            # Indicate that the algorithm was fully run.
            results.run_algorithm = True
        if verbose:
            bar.finish()
        # Store the time spent to run the algorithm.
        results.algorithm_running_time = time.time() - init_time
        return results


class AlgorithmResults:

    """An AlgorithmResults object is used to store the output generated by the
    run_algorithm method. 
    
    """

    def __init__(self, data, budget):
        """Initiate an AlgorithmResults object. """
        # Store the budget and the Data object associated with the
        # AlgorithmResults object.
        self.data = data
        self.budget = budget
        # Compute the initial state and the first best state.
        self.init_state = np.zeros(self.data.individuals, dtype=int)
        self.last_state = np.array(self.data.alternatives_per_individual) - 1
        # Compute initial utility, first best utility and max
        # budget necessary. 
        self.init_utility = self.data._total_utility(self.init_state)
        self.last_utility = self.data._total_utility(self.last_state)
        self.max_expenses = self.init_utility - self.last_utility
        self.percent_max_expenses = self.budget / self.max_expenses
        # Same for energy.
        self.init_energy = self.data._total_energy(self.init_state)
        self.last_energy = self.data._total_energy(self.last_state)
        self.max_total_energy_gains = self.init_energy - self.last_energy
        # The variable optimal_state is a numpy array with the alternative
        # chosen by all the individuals at the optimal state, the array is 
        # initialized with all individuals being at their first alternative.
        self.optimal_state = np.zeros(data.individuals, dtype=int)
        # The variable iteration counts the number of iterations of the
        # algorithm.
        self.iteration = 0
        # The variable expenses is equal to the total amount of incentives
        # needed to reach the optimal state.
        self.expenses = 0
        # The variable total_energy_gains is equal to the decrease in energy
        # consumption at the optimal state compared to the initial state.
        self.total_energy_gains = 0
        # At each iteration, the selected individual, the previous alternative 
        # and the next alternative are stored in the variable jumps_history.
        # There are as many rows as iterations.
        # There are three columns: selected individual, previous jump and next
        # jump.
        self.jumps_history = []
        # The variable incentives_history is a list where the amount of
        # incentives for each iteration is stored.
        self.incentives_history = []
        # The variable energy_gains_history is a list where the amount of
        # energy gains for each iteration is stored.
        self.energy_gains_history = []
        # The variable efficiencies_history is a list where the efficiency of
        # each jump is stored.
        self.efficiencies_history = []
        # The variable overshot_jump is used to store information on the
        # overshot jump (the jump that is removed in case of overshot).
        self.overshot_jump = None
        # The variable run_algorithm is True if the algorithm was fully run.
        self.run_algorithm = False
        # The variable computed_results is True if the method compute_results
        # was run.
        self.computed_results = False
        # Generate empty variables to store the computing times.
        self.algorithm_running_time = None
        self.computing_results_time = None
        self.output_results_time = None
        self.output_characteristic_time = None

    def compute_results(self, verbose=True):
        """Compute relevant results from the raw results of the algorithm.

        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert self.run_algorithm, \
                'The algorithm was not run.'
        if verbose:
            print('Computing additional results...')
        # Compute the incentives needed to move the individuals from their
        # first alternative to their optimal alternative.
        self.total_incentives = [
            self.data._incentives_amount(i, 0, self.optimal_state[i]) 
            for i in range(self.data.individuals)
        ]
        # Compute optimal utility and optimal energy gains.
        self.optimal_utility = self.init_utility - self.expenses
        self.optimal_energy = self.init_energy - self.total_energy_gains
        # Compute the remaining budget and the percentage of budget used.
        self.budget_remaining = self.budget - self.expenses
        self.percent_budget = self.expenses / self.budget
        # Compute the energy gains as percentage of total energy gains possible.
        self.percent_total_energy_gains = \
            self.total_energy_gains / self.max_total_energy_gains
        # Compute efficiency of first and last jump and total efficiency.
        self.first_efficiency = self.efficiencies_history[0]
        self.last_efficiency = self.efficiencies_history[-1]
        self.optimal_state_efficiency = self.total_energy_gains / self.expenses
        # Compute the maximum, minimum and average amount of incentives of a
        # jump.
        self.max_incentives = max(self.incentives_history)
        self.min_incentives = min(self.incentives_history)
        self.average_incentives = np.mean(self.incentives_history)
        # Compute the maximum, minimum and average energy gains of a jump.
        self.max_energy_gains = max(self.energy_gains_history)
        self.min_energy_gains = min(self.energy_gains_history)
        self.average_energy_gains = np.mean(self.energy_gains_history)
        # Compute the average number of jumps (= iterations).
        self.average_nb_jumps = self.iteration / self.data.individuals
        # Compute the percentage of individuals at first and last alternative.
        self.percent_at_first_alternative = np.sum(self.optimal_state==0) \
                                            / self.data.individuals
        self.percent_at_last_alternative = np.sum(self.optimal_state \
            == np.array(self.data.alternatives_per_individual)-1) \
            / self.data.individuals
        # Compute the bound interval and the bound.
        if self.overshot_jump is None:
            self.bound_size = 0
        else:
            self.bound_size = self.data._energy_gains_amount(
                                    self.overshot_jump[0],
                                    self.overshot_jump[1],
                                    self.overshot_jump[2]
                                )
        self.max_bound = self.total_energy_gains
        self.min_bound = self.max_bound - self.bound_size
        # Compute the amount of expenses at each iteration.
        self.expenses_history = \
            np.append(0, np.cumsum(self.incentives_history))
        # Compute the total energy gains at each iteration.
        self.total_energy_gains_history = \
            np.append(0, np.cumsum(self.energy_gains_history))
        # Compute the bound differences.
        self.bound_differences = \
            np.append(np.diff(self.total_energy_gains_history), 0)
        # Compute an array going from 0 to the number of iterations.
        self.iterations_history = np.arange(0, self.iteration, 1)
        # To compute the number of individuals at their first alternative for 
        # each iteration, we build an array where each element is 1 if the 
        # previous alternative of the associated jump is 0 and else is 0. The 
        # cumulative sum of this array gives the number of individuals NOT at 
        # their first alternative. The number of individuals at their first 
        # alternative is the total number of individuals minus the array we 
        # computed.
        x = np.zeros(self.iteration, dtype=int)
        y = np.array(self.jumps_history)[:,1]
        x[y==0] = 1
        self.moved_history = np.cumsum(x)
        self.at_first_alternative_history = \
            self.data.individuals - self.moved_history
        # To compute the number of individuals at their last alternative, we 
        # use the same method but the elements of the first array is 1 if the 
        # next alternative of the jump is equal to the last jump of the 
        # associated individual. We also need to add the number of individuals 
        # with one alternative (they are always at their last alternative).
        x = np.zeros(self.iteration, dtype=int)
        y = np.array(self.jumps_history)[:,2]
        z = np.array(self.jumps_history)[:,0]
        z = [self.last_state[i] for i in z]
        x[y==z] = 1
        x = np.cumsum(x)
        w = np.sum(self.last_state==0)
        self.at_last_alternative_history = x + w
        # Inform that the results were computed.
        self.computed_results = True
        # Store the time spent computing results.
        self.computing_results_time = time.time() - init_time

    def output_characteristics(self, filename, verbose=True):
        """Write a file with some characteristics on the results of the algorithm.

        :filename: string with the name of the file where the information are 
        written
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert self.run_algorithm, \
                'The algorithm was not run.'
        # Try to open file filename and return FileNotFoundError if python is 
        # unable to open the file.
        try:
            output_file = open(filename, 'w')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        # Run the method compute_results if it was not already done.
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Writing some characteristics about the results on a file...')
        size = 60
        # Display the budget, the expenses, the percentage of budget spent
        # and the remaining budget.
        output_file.write('Expenses'.center(size-1, '='))
        output_file.write('\nBudget:'.ljust(size) + "{:,}".format(self.budget))
        output_file.write('\nAmount spent:'.ljust(size) 
                          + "{:,.4f}".format(self.expenses))
        output_file.write('\nRemaining budget:'.ljust(size) 
                          + "{:,.4f}".format(self.budget_remaining))
        output_file.write('\nPercentage of budget spent:'.ljust(size) 
                          + "{:,.2%}".format(self.percent_budget))
        output_file.write('\n\n')
        # Display the initial total individual utility and the total individual
        # utility at optimal state.
        output_file.write('Individual Utility'.center(size-1, '='))
        output_file.write('\nInitial total individual utility:'.ljust(size) 
                          + "{:,.4f}".format(self.init_utility))
        output_file.write('\nOptimal total individual utility:'.ljust(size) 
                          + "{:,.4f}".format(self.optimal_utility))
        output_file.write('\n\n')
        # Display the initial energy consumption, the energy consumption at
        # optimal state and the energy gains.
        output_file.write('Energy Consumption'.center(size-1, '='))
        output_file.write('\nInitial energy consumption:'.ljust(size) 
                          + "{:,.4f}".format(self.init_energy))
        output_file.write('\nEnergy gains:'.ljust(size) 
                          + "{:,.4f}".format(self.total_energy_gains))
        output_file.write('\nOptimal energy consumption:'.ljust(size) 
                          + "{:,.4f}".format(self.optimal_energy))
        output_file.write('\n\n')
        # Display information on the distance between the optimal state and the
        # last state (first best).
        output_file.write('Distance from First Best'.center(size-1, '='))
        output_file.write('\nMaximum budget necessary:'.ljust(size) 
                          + "{:,.4f}".format(self.max_expenses))
        output_file.write(('\nActual budget in percentage of maximum budget '
                          + 'necessary:').ljust(size) 
                          + "{:,.2%}".format(self.percent_max_expenses))
        output_file.write('\nMaximum energy gains:'.ljust(size) 
                          + "{:,.4f}".format(self.max_total_energy_gains))
        output_file.write(('\nActual energy gains in percentage of maximum '
                          + 'energy gains:').ljust(size) 
                          + "{:,.2%}".format(self.percent_total_energy_gains))
        output_file.write('\n\n')
        # Display the efficiency of the first jump and the last jump and the
        # efficiency of the optimal state.
        output_file.write('Efficiency'.center(size-1, '='))
        output_file.write('\nEfficiency of first jump:'.ljust(size) 
                          + "{:,.4f}".format(self.first_efficiency))
        output_file.write('\nEfficiency of last jump:'.ljust(size) 
                          + "{:,.4f}".format(self.last_efficiency))
        output_file.write('\nEfficiency of last state:'.ljust(size) 
                          + "{:,.4f}".format(self.optimal_state_efficiency))
        output_file.write('\n\n')
        # Display the maximum, minimum and average amount of incentives and
        # energy gains.
        output_file.write('Jumps'.center(size-1, '='))
        output_file.write('\nLargest amount of incentives:'.ljust(size) 
                          + "{:,.4f}".format(self.max_incentives))
        output_file.write('\nSmallest amount of incentives:'.ljust(size) 
                          + "{:,.4f}".format(self.min_incentives))
        output_file.write('\nAverage amount of incentives:'.ljust(size) 
                          + "{:,.4f}".format(self.average_incentives))
        output_file.write('\nLargest energy gains of a jump:'.ljust(size) 
                          + "{:,.4f}".format(self.max_energy_gains))
        output_file.write('\nSmallest energy gains of a jump:'.ljust(size) 
                          + "{:,.4f}".format(self.min_energy_gains))
        output_file.write('\nAverage energy gains of a jump:'.ljust(size) 
                          + "{:,.4f}".format(self.average_energy_gains))
        output_file.write('\n\n')
        # Display the total number of jumps, the average number of jumps per 
        # individual, the percentage of individuals that did not moved and the 
        # percentage of individuals at their last alternative.
        output_file.write('Individuals and Jumps'.center(size-1, '='))
        output_file.write('\nTotal number of jumps:'.ljust(size) 
                          + "{:,}".format(self.iteration))
        output_file.write(('\nAverage number of jumps per '
                          + 'individual:').ljust(size) 
                          + "{:,.4f}".format(self.average_nb_jumps))
        output_file.write(('\nPercentage of individuals that did not '
                          + 'moved:').ljust(size) 
                          + "{:,.2%}".format(self.percent_at_first_alternative))
        output_file.write(('\nPercentage of individuals at their last '
                          + 'alternative:').ljust(size) 
                          + "{:,.2%}".format(self.percent_at_last_alternative))
        output_file.write('\n\n')
        # Display the interval of consumption energy at optimum.
        output_file.write('Bound'.center(size-1, '='))
        output_file.write('\nMinimum energy consumption at optimum:'.ljust(size)
                          + "{:,.4f}".format(self.min_bound))
        output_file.write('\nMaximum energy consumption at optimum:'.ljust(size)
                          + "{:,.4f}".format(self.max_bound)) 
        output_file.write('\nBound:'.ljust(size) 
                          + "{:,.4f}".format(self.bound_size))
        output_file.write('\n\n')
        # Display computation times.
        output_file.write('Technical Information'.center(size-1, '='))
        if not self.data.read_time is None:
            output_file.write('\nTime to read the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.read_time))
        if not self.data.generating_time is None:
            output_file.write('\nTime to generate the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.generating_time))
        if not self.data.sorting_time is None:
            output_file.write('\nTime to sort the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.sorting_time))
        if not self.data.pareto_removing_time is None:
            output_file.write(('\nTime to remove Pareto-dominated'
                               + ' alternatives (s):').ljust(size)
                              + "{:,.4f}".format(self.data.pareto_removing_time))
        if not self.data.output_data_time is None:
            output_file.write('\nTime to output the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.output_data_time))
        if not self.data.output_characteristics_time is None:
            output_file.write(('\nTime to output characteristics on the data '
                              + '(s):').ljust(size)
                              + "{:,.4f}".format(
                                  self.data.output_characteristics_time))
        if not self.algorithm_running_time is None:
            output_file.write('\nTime to run the algorithm (s):'.ljust(size)
                              + "{:,.4f}".format(self.algorithm_running_time))
        if not self.computing_results_time is None:
            output_file.write('\nTime to compute additional results (s):'.ljust(
                              size)
                              + "{:,.4f}".format(self.computing_results_time))
        if not self.output_results_time is None:
            output_file.write('\nTime to output the results (s):'.ljust(size)
                              + "{:,.4f}".format(self.output_results_time))
        # Store the time spent to output characteristics.
        self.output_characteristics_time = time.time() - init_time
        output_file.write(('\nTime to output characteristics on the results '
                          + '(s):').ljust(size)
                          + "{:,.4f}".format(self.output_characteristics_time))

    def output_results(self, filename, verbose=True):
        """Write a file with the results of the algorithm.

        The results include the alternative chosen by each individual at 
        optimum and the amount of incentives to give to each one.

        :filename: string with the name of the file where the results are 
        written
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert self.run_algorithm, \
                'The algorithm was not run.'
        # Try to open file filename and return FileNotFoundError if python is 
        # unable to open the file.
        try:
            output_file = open(filename, 'w')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Writing the results of the algorithm on a file...')
        # The first row is the label of the three columns.
        output_file.write('Individual | Alternative | Incentives\n')
        # There is one row for each individual.
        for i in range(self.data.individuals):
            output_file.write(
                str(i+1).center(10)
                + ' |'
                + str(self.optimal_state[i]).center(13)
                + '| '
                + str(self.total_incentives[i]).center(10)
                + '\n'
            )
        # Store the time spent to output results.
        self.output_results_time = time.time() - init_time

    def plot_efficiency_curve(self, filename=None, verbose=True):
        """Plot the efficiency curve with the algorithm results.

        The efficiency curve relates the expenses with the energy gains.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the efficiency curve...')
        _plot_step_function(
                self.expenses_history,
                self.total_energy_gains_history,
                title='Efficiency Curve', 
                xlabel='Expenses', 
                ylabel='Energy gains', 
                filename=filename
        )

    def plot_efficiency_evolution(self, filename=None, verbose=True):
        """Plot the evolution of the jump efficiency over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the jump efficiency...')
        _plot_scatter(
                self.iterations_history,
                self.efficiencies_history,
                'Evolution of the Efficiency of the Jumps',
                'Iterations',
                'Efficiency',
                regression=False,
                filename=filename
        )

    def plot_incentives_evolution(self, filename=None, verbose=True):
        """Plot the evolution of the jump incentives over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the jump incentives...')
        _plot_scatter(
                self.iterations_history,
                self.incentives_history,
                'Evolution of the Incentives of the Jumps',
                'Iterations',
                'Incentives',
                regression=False,
                filename=filename
        )

    def plot_energy_gains_evolution(self, filename=None, verbose=True):
        """Plot the evolution of the jump energy gains over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the jump energy gains...')
        _plot_scatter(
                self.iterations_history,
                self.energy_gains_history,
                'Evolution of the Energy Gains of the Jumps',
                'Iterations',
                'Energy gains',
                regression=False,
                filename=filename
        )
        
    def plot_bounds(self, filename=None, bounds=True, differences=True, 
            verbose=True):
        """Plot the lower and upper bounds of the total energy gains for each
        level of expenses. Also plot the difference between the lower and upper
        bounds.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :bounds: if True, plot the lower and upper bounds, default is True
        :difference: if True, plot the bound differences, default is True
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the bounds of the total energy gains...')
        # Initiate the graph.
        fig, ax = plt.subplots()
        # The x-coordinates are the expenses history.
        x = self.expenses_history
        # The y-coordinates are the total energy gains history.
        y = self.total_energy_gains_history
        if bounds:
            # The upper bounds are an offset efficiency curve.
            ax.step(x, self.total_energy_gains_history, 'g', where='pre', 
                    label='Upper bounds')
            # The lower bounds are the efficiency curve.
            ax.step(x, self.total_energy_gains_history, 'r', where='post', 
                    label='Lower bounds')
        if differences:
            # Plot the bound differences.
            ax.step(x, self.bound_differences, 'b', where='post', 
                    label='Bound differences')
        # Add the title and the axis label.
        ax.set_title('Energy gains bounds')
        ax.set_xlabel('Expenses')
        ax.set_ylabel('Energy gains')
        # Display a legend.
        plt.legend()
        # Make room for the labels.
        plt.tight_layout()
        # Show the graph if no file is specified.
        if filename is None:
            plt.show()
        # Save the graph as a png file if a file is specified.
        else:
            plt.savefig(filename, format='png')
            plt.close()

    def plot_individuals_who_moved(self, filename=None, verbose=True):
        """Plot the evolution of the number of individuals who moved over the
        iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the number of individuals who moved...')
        _plot_step_function(
                self.iterations_history,
                self.moved_history,
                title='Number of Individuals who Moved',
                xlabel='Iterations',
                ylabel='Number of individuals',
                filename=filename
        )

    def plot_individuals_at_first_best(self, filename=None, verbose=True):
        """Plot the evolution of the number of individuals at their last
        alternative over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print(('Plotting the evolution of the number of individuals at '
                  + 'first best...'))
        _plot_step_function(
                self.iterations_history,
                self.at_last_alternative_history,
                title='Number of Individuals at First Best Alternative',
                xlabel='Iterations',
                ylabel='Number of individuals',
                filename=filename
        )


class Regression:

    """A Regression object stores information on the regression between two
    variables. 
    """

    def __init__(self, x, y):
        """Initiate variables, perform a regression and create a legend. """
        self.x = x
        self.y = y
        self.degree = 3
        # Do a polynomial regression.
        self._polynomial_regression()
        # Create a legend.
        self._legend(4)

    def _polynomial_regression(self):
        """Compute the coefficients, the covariance matrix, the t-statistics 
        and the significance (boolean) of a polynomial regression.
        """
        # Compute the coefficients and the covariance matrix of the regression
        # for a polynomial with 3 degrees.
        self.coefficients, self.covariance = np.polyfit(self.x, 
                                                        self.y, 
                                                        self.degree, 
                                                        cov=True)
        # Compute the statistical significance of the coefficients (estimate
        # divided by its standard error).
        t_statistics = []
        for i in range(self.degree+1):
            t = abs(self.coefficients[i]) / self.covariance[i, i]**(1/2)
            t_statistics.append(t)
        # The coefficients are statistically significant if the t-statistic is
        # greater than 1.96.
        t_statistics = np.array(t_statistics)
        self.significance = t_statistics > 1.96
        # Store the number of significant coefficients.
        self.nb_significant = sum(self.significance)

    def _r_squared(self):
        """Compute the R² of the regression.
        """
        # Compute the predicted values of y using only the significant
        # coefficients.
        self.poly = np.poly1d(self.coefficients)
        y_hat = self.poly(self.x)
        # Compute the mean of y.
        y_bar = sum(self.y)/len(self.y)
        y_bars = np.repeat(y_bar, len(self.y))
        # Compute the sum of squared residuals and the total sum of squares.
        SSR = np.sum((y_hat - self.y)**2)
        SST = np.sum((y_bars - self.y)**2)
        # Compute the R².
        self.R2 = SSR / SST

    def _legend(self, r):
        """Create a string with information on the regression that can be
        displayed through the legend of a plot.

        :r: precision of round

        """
        if self.nb_significant == 0:
            self.legend = '$y$ = 0\n$R^2$=0'
            self.poly = np.poly1d(0)
        else:
            # Create a string with the equation of the regression line.
            equation = '$y$ = '
            # Store the value of the significant coefficients and the associated
            # degree of x.
            significant_coefficients = [
                                        (self.degree-i, c) for i, c 
                                        in enumerate(self.coefficients) 
                                        if self.significance[i]
                                       ]
            # Add the value of the significant coefficient with the higher degree.
            equation += str(round(significant_coefficients[0][1], r))
            # Add x with its associated degree.
            equation += ' ' + self._x_string(significant_coefficients[0][0])
            # For the other significant coefficients (if they exist), add their
            # value and the x associated.
            for i in range(self.nb_significant-1):
                # Add the sign of the coefficient.
                if significant_coefficients[i+1][1] > 0:
                    equation += ' + '
                else:
                    equation += ' - '
                # Add the absolute value of the coefficient.
                equation += str(abs(round(significant_coefficients[i+1][1], r)))
                # Add the x with its associated degree.
                equation += ' ' + self._x_string(significant_coefficients[i+1][0])
            # Compute the R².
            self._r_squared()
            # The first line of the string is the equation and the second line is
            # the R².
            self.legend = equation + '\n$R^2$ = ' + str(round(self.R2, r))

    def _x_string(self, deg):
        """Create a string with x and a specified exponent.

        For instance, if degree is d>1, return '$x^d$'.
        If the exponent is 1, return '$x$'.
        If the exponent is 0, return ''.

        :degree: degree associated, must be a int
        :returns: a string

        """
        if deg == 0:
            s = ''
        elif deg == 1:
            s = r'$x$'
        elif deg >= 2:
            s = r'$x^' + str(deg) + '$'
        return s
        

###############
#  Functions  #
###############


def _known_custom_bar(max_length, text):
    """Create a progress bar of specified length.

    :max_length: max length of the progress bar, should be an integer
    :text: string printed in the middle of the progress bar, should be
    string
    :returns: ProgressBar object

    """
    blanks = 25 - len(text)
    bar = progressbar.ProgressBar(
            max_value=max_length,
            widgets=[
                text,
                '... ',
                blanks * ' ',
                progressbar.Timer(), 
                ' ',
                progressbar.ETA(),
                ' ',
                progressbar.Bar(left='[', right=']', fill='-'), 
                progressbar.Percentage()
            ])
    return bar


def _unknown_custom_bar(main_text, counter_text):
    """Create a progress bar of unknown length.

    :main_text: string printed in the middle of the progress bar, should be
    string
    :counter_text: string printed after the counter indicator, should be string
    :returns: ProgressBar object and counter integer

    """
    blanks = 25 - len(main_text)
    bar = progressbar.ProgressBar(
            max_value=progressbar.UnknownLength,
            widgets=[
                main_text,
                '... ',
                blanks * ' ',
                progressbar.Timer(), 
                ' (', 
                progressbar.Counter(),
                ' ',
                counter_text,
                ')',
                ' ',
                progressbar.AnimatedMarker()
            ])
    counter = 1
    return bar, counter


def _plot_step_function(x, y, title, xlabel, ylabel, filename=None):
    """Plot a step function.

    :x: list or numpy array with the x-coordinates
    :y: list or numpy array with the y-coordinates
    :title: title of the graph
    :xlabel: label of the x-axis
    :ylabel: label of the y-axis
    :file: string with the name of the file where the graph is saved, if
    None show the graph but does not save it, default is None

    """
    # Initiate the graph.
    fig, ax = plt.subplots()
    # Plot the line.
    ax.step(x, y, 'b', where='post')
    # Add the title and the axis label.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Make room for the labels.
    plt.tight_layout()
    # Show the graph if no file is specified.
    if filename is None:
        plt.show()
    # Save the graph as a png file if a file is specified.
    else:
        plt.savefig(filename, format='png')
        plt.close()


def _plot_scatter(x, y, title, xlabel, ylabel, regression=True, filename=None):
    """Plot a scatter.

    :x: list or numpy array with the x-coordinates
    :y: list or numpy array with the y-coordinates
    :title: title of the graph
    :xlabel: label of the x-axis
    :ylabel: label of the y-axis
    :regression: if true, perform a regression and display the regression line
    and a legend
    :file: string with the name of the file where the graph is saved, if
    None show the graph but does not save it, default is None

    """
    # Initiate the graph.
    fig, ax = plt.subplots()
    # Plot the scatter.
    ax.scatter(x, y, s=5)
    # Perform a regression if necessary.
    if regression:
        reg = Regression(x, y)
        xs = np.linspace(*ax.get_xlim(), 1000)
        ys = reg.poly(xs)
        ax.plot(xs, ys, color='red', label=reg.legend)
        ax.legend()
    # Do not show negative values on the y-axis if all values are positive.
    if ax.get_ylim()[0] < 0 and min(y) >= 0:
        ax.set_ylim(bottom=0)
    # Add the title and the axis label.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Make room for the labels.
    plt.tight_layout()
    # Show the graph if no file is specified.
    if filename is None:
        plt.show()
    # Save the graph as a png file if a file is specified.
    else:
        plt.savefig(filename, format='png')
        plt.close()


def _simulation(budget=np.infty, rem_eff=True, 
        verbose=True, **kwargs):
    """Generate random data and run the algorithm.

    To specify the parameters for the generation process, use the same syntax as
    for the method Data.generate().

    :budget: budget used to run the algorithm, by default budget is infinite
    :rem_eff: if True, the efficiency dominated alternatives
    are removed before the algorithm is run
    :verbose: if True, display progress bars and some information
    :returns: an AlgorithmResults object with the results of the algorithm run

    """
    # Create a Data object.
    data = Data()
    # Generate random data.
    data.generate(verbose=verbose, **kwargs)
    if rem_eff:
        # Remove the efficiency dominated alternatives.
        data.remove_efficiency_dominated(verbose=verbose)
    # Run the algorithm.
    results = data.run_algorithm(budget=budget, verbose=verbose)
    return results


def _run_algorithm(simulation=None, filename=None, budget=np.infty,
        remove_efficiency_dominated = True, directory='files', delimiter=',', 
        comment='#', verbose=True, **kwargs):
    """Run the algorithm and generate files and graphs.

    The algorithm can be run with generated data or with imported data.

    :simulation: boolean indicated whether the data must be generated or 
    imported
    :filename: string with the name of the file containing the data
    :budget: budget used to run the algorithm, by default budget is infinite
    :remove_efficiency_dominated: if True, the efficiency dominated alternatives
    are removed before the algorithm is run
    :directory: directory where the files are stored, must be a string, default
    is 'files'
    :delimiter: the character used to separated the utility and the energy
    consumption of the alternatives, default is comma
    :comment: line starting with this string are not read, should be a
    string, default is #
    :verbose: if True, display progress bars and some information

    """
    # Store the starting time.
    init_time = time.time()
    # Create the directory used to store the files.
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    if simulation:
        # Run the simulation.
        results = _simulation(budget=budget,
                              rem_eff=remove_efficiency_dominated, 
                              verbose=verbose, 
                              **kwargs)
    else:
        # Import the data.
        data = Data()
        data.read(filename, delimiter=delimiter, comment=comment, verbose=verbose)
        # Run the algorithm.
        results = data.run_algorithm(budget=budget, verbose=verbose)
    # Generate the files and the graphs.
    results.data.output_data(filename=directory+'/data.txt', verbose=verbose)
    results.data.output_characteristics(
            filename=directory+'/data_characteristics.txt',
            verbose=verbose
            )
    results.output_results(
            filename=directory+'/results.txt', 
            verbose=verbose
            )
    results.output_characteristics(
            filename=directory+'/results_characteristics.txt', 
            verbose=verbose
            )
    results.plot_efficiency_curve(
            filename=directory+'/efficiency_curve.png',
            verbose=verbose
            )
    results.plot_efficiency_evolution(
            filename=directory+'/efficiency_evolution.png',
            verbose=verbose
            )
    results.plot_incentives_evolution(
            filename=directory+'/incentives_evolution.png',
            verbose=verbose
            )
    results.plot_energy_gains_evolution(
            filename=directory+'/energy_gains_evolution.png',
            verbose=verbose
            )
    results.plot_bounds(
            filename=directory+'/bounds.png', 
            verbose=verbose
            )
    results.plot_individuals_who_moved(
            filename=directory+'/individuals_who_moved.png',
            verbose=verbose
            )
    results.plot_individuals_at_first_best(
            filename=directory+'/individuals_at_first_best.png',
            verbose=verbose
            )
    # Store the total time to run the simulation.
    total_time = time.time() - init_time
    if verbose:
        print('Finished! (Elapsed Time: ' 
              + str(round(total_time, 2)) 
              + 's)')


def run_simulation(budget=np.infty, directory='files', verbose=True, **kwargs):
    """Create files and graphs while generating random data and running the 
    algorithm.

    To specify the parameters for the generation process, use the same syntax as
    for the method Data.generate().
    The generated files are the data, data characteristics, the results and results
    characteristics.
    The generated graphs are efficiency curve, efficiency evolution, incentives
    evolution, energy gains evolution, bounds, individuals who moved and
    individuals at first best.

    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: directory where the files are stored, must be a string, default
    is 'files'
    :verbose: if True, display progress bars and some information

    """
    _run_algorithm(simulation=True, budget=budget, directory=directory, verbose=verbose,
            **kwargs)


def run_from_file(filename, budget=np.infty, directory='files', delimiter=',', 
        comment='#', verbose=True):
    """Read data from a file and run the algorithm.

    The generated files are the data, data characteristics, the results and results
    characteristics.
    The generated graphs are efficiency curve, efficiency evolution, incentives
    evolution, energy gains evolution, bounds, individuals who moved and
    individuals at first best.

    :filename: string with the name of the file containing the data
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: directory where the files are stored, must be a string, default
    is 'files'
    :delimiter: the character used to separated the utility and the energy
    consumption of the alternatives, default is comma
    :comment: line starting with this string are not read, should be a
    string, default is #
    :verbose: if True, display progress bars and some information

    """
    _run_algorithm(simulation=False, filename=filename, budget=budget, directory=directory,
            delimiter=delimiter, comment=comment, verbose=verbose)


def _complexity(varying_parameter, string, start, stop, step, budget=np.infty, 
        directory='complexity', verbose=True, **kwargs):
    """Run multiple simulations with a parameter varying and compute time
    complexity of the algorithm.

    :varying_parameter: string specifying the parameter which varies across
    simulations, possible values are 'individuals', 'alternatives' and 'budget'
    :string: string with the name of the varying parameter, used to label the
    graphs
    :start: start value for the interval of number of individuals
    :stop: end value for the interval of number of individuals, this value is
    not include in the interval
    :step: spacing between values in the interval
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: string specifying the directory where the files are stored
    :verbose: if True, a progress bar and some information are displayed during

    """
    # Check that varying_parameter is well specified.
    assert varying_parameter in ['individuals', 'alternatives', 'budget'], \
        'The varying parameter is not well specified'
    # Create the directory used to store the files.
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    # Compute the interval of values for the number of individuals.
    X = np.arange(start, stop, step, dtype=int)
    if verbose:
        # Print a progress bar of duration the number of simulations.
        bar = _known_custom_bar(len(X), 'Running simulations')
    # Generate empty lists to store the computing times.
    generating_times = []
    pareto_removing_times = []
    running_times = []
    # Run a simulation for each value in the interval and store relevant
    # results.
    for i, x in enumerate(X):
        if verbose:
            bar.update(i)
        data = Data()
        time0 = time.time()
        # Generate the data.
        if varying_parameter == 'individuals':
            data.generate(individuals=x, verbose=False, **kwargs)
        elif varying_parameter == 'alternatives':
            data.generate(mean_nb_alternatives=x, verbose=False, **kwargs)
        elif varying_parameter == 'budget':
            data.generate(verbose=False, **kwargs)
        time1 = time.time()
        generating_times.append(time1 - time0)
        # Remove the Pareto dominated alternatives.
        data.remove_pareto_dominated(verbose=False)
        time2 = time.time()
        pareto_removing_times.append(time2 - time1)
        # Run the algorithm.
        if varying_parameter == 'budget':
            data.run_algorithm(budget=x, verbose=False)
        else:
            data.run_algorithm(budget=budget, verbose=False)
        time3 = time.time()
        running_times.append(time3 - time2)
    bar.finish()
    # Plot graphs showing time complexity.
    _plot_scatter(
            X, 
            generating_times, 
            'Time Complexity with the '+string+ '\n(Generating Time)', 
            string, 
            'Generating Time',
            filename=directory+'/generating_time.png'
            )
    _plot_scatter(
            X, 
            pareto_removing_times, 
            'Time Complexity with the '+string 
            + '\n(Time to Remove Pareto-Dominated Alternatives)',
            string, 
            'Time to Remove Pareto-Dominated Alternatives',
            filename=directory+'/removing_pareto_dominated_times.png'
            )
    _plot_scatter(
            X, 
            running_times, 
            'Time Complexity with the '+string + '\n(Running Time)',
            string, 
            'Running Time',
            filename=directory+'/running_time.png'
            )
    if verbose:
        print('Successfully run ' + str(len(X)) + ' simulations.')


def complexity_individuals(start, stop, step, budget=np.infty,
        directory='complexity_individuals', verbose=True, **kwargs):
    """Run multiple simulations with a varying number of individuals and compute
    time complexity of the algorithm.

    To specify the parameters for the generation process, use the same syntax as
    for the method Data.generate().

    :start: start value for the interval of number of individuals
    :stop: end value for the interval of number of individuals, this value is
    not include in the interval
    :step: spacing between values in the interval
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: string specifying the directory where the files are stored
    :verbose: if True, a progress bar and some information are displayed during
    the process, default is True

    """
    string = 'Number of Individuals'
    _complexity('individuals', string, start, stop, step, budget=budget,
            directory=directory, verbose=verbose, **kwargs)


def complexity_alternatives(start, stop, step, budget=np.infty, 
        directory='complexity_alternatives', verbose=True, **kwargs):
    """Run multiple simulations with a varying average number of alternatives 
    and compute time complexity of the algorithm.

    To specify the parameters for the generation process, use the same syntax as
    for the method Data.generate().

    :start: start value for the interval of average number of alternatives
    :stop: end value for the interval of average number of alternatives, this 
    value is not include in the interval
    :step: spacing between values in the interval
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: string specifying the directory where the files are stored
    :verbose: if True, a progress bar and some information are displayed during
    the process, default is True

    """
    string = 'Average Number of Alternatives'
    _complexity('alternatives', string, start, stop, step, budget=budget,
            directory=directory, verbose=verbose, **kwargs)


def complexity_budget(start, stop, step, directory='complexity_budget',
        verbose=True, **kwargs):
    """Run multiple simulations with a varying budget and compute time 
    complexity of the algorithm.

    To specify the parameters for the generation process, use the same syntax as
    for the method Data.generate().

    :start: start value for the interval of budget
    :stop: end value for the interval of budget, this value is not include in the 
    interval
    :step: spacing between values in the interval
    :directory: string specifying the directory where the files are stored
    :verbose: if True, a progress bar and some information are displayed during
    the process, default is True

    """
    string = 'Budget'
    _complexity('budget', string, start, stop, step, directory=directory, 
            verbose=verbose, **kwargs)
