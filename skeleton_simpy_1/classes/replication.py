from classes.model import Model

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

pd.set_option('display.max_rows', 120)
pd.set_option('display.max_columns', 120)


class Replicator:
    
    """
    Replicator class. Calls multiple replicates of multiple scenarios.
    Summarises data using Pandas DataFrames

    Sequence of calls:
    ------------------
        1. `run_scenarios` is called from oustide method
        2. `run_scenarios` calls `run_trial` for each sceanrio
        3. `run_trial` calls `single_run` for each replicate run. This is 
            performed across multiple CPU cores.
        4. `collate_trial_results` collates replicate trial runs for a single
            scenario and groups them in one DataFrame.
        5. `pivot_results` takes the collated DataFrames (one per sceanrio) and
            produces summary results.
        6. When all scenarios and trials are run, `run_scenarios` will print and
            save results.
        """

    def __init__(self, scenarios, replications):
        """Constructor method for replicator."""
        
        self.replications = replications
        self.scenarios = scenarios
        
        # Set up dictionary for concatenated results
        self.results = dict()


    def aggregate_results(self):

        # Define functions to allow percentiles in a Pandas Pivot Table
        def percent_5(g):
            return np.percentile(g, 5)

        def percent_95(g):
            return np.percentile(g, 75)

        self.aggregated_results = dict()

        # Loop through scenarios
        for scenario_name in self.scenarios.keys():
            # Loop through results tables for each scenario
            for results_name in self.results_table_names:
                # Get results and drop name and run #
                df = self.trial_results[scenario_name, results_name]
                df.drop(['name', 'run'], axis=1, inplace=True)
                # Average by index and store
                grouped = df.groupby(
                    by=df.index.name).agg([percent_5, np.median, percent_95])
                self.aggregated_results[(scenario_name, results_name)] = grouped


    def print_and_save_results(self):

        # Get indexes (extract from tuples)
        tuple_indexes = self.aggregated_results.keys()
        scenarios = []
        result_names = []
        for index in tuple_indexes:
            scenarios.append(index[0])
            result_names.append(index[1])
        scenarios = list(set(scenarios)); scenarios.sort()
        result_names = list(set(result_names)); result_names.sort()

        # Print and save
        for scenario in scenarios:
            for result_name in result_names:
                df = self.aggregated_results[scenario, result_name]
                # Save full results
                df.to_csv(f'./skeleton_simpy_1/output/{scenario}_{result_name}.csv')

                # Display median results only (and remove 'median' as column name)
                display_cols = []
                for col in df.columns.values.tolist():
                   if col[-1] == 'median':
                       display_cols.append(col)
                df.columns = [col[0:-1] for col in df.columns]

                print (f'Scenario: {scenario}, Result: {result_name}')
                print(df, end='\n\n')



        pass


    def run_scenarios(self):
        """Calls for replications of each scenario, calls for summarisation, 
        displaying of results, and saving of results."""
        
        # Run all named scenarios
        scenario_count = len(self.scenarios)
        counter = 0
        self.trial_results = dict()
        for name, scenario in self.scenarios.items():
            
            # Run trial
            counter += 1
            print(f'\r>> Running scenario {counter} of {scenario_count}', end='')
            # Call for all replications of a single scenario to be run
            results = self.run_trial(scenario)

            # Collate
            self.results_table_names = []
            for results_name in results[0].keys():
                self.results_table_names.append(results_name)
                results_list = []
                for i in range(len(results)):
                    results[i][results_name]['run'] = i + 1
                    results[i][results_name]['name'] = name
                    results_list.append(results[i][results_name])

                self.trial_results[(name, results_name)] = \
                    pd.concat(results_list, axis=0)

        # Clear displayed progress output
        clear_line = '\r' + " " * 79
        print(clear_line, end = '')
        
        # Pivot results (Get summary results for all scenarios)
        self.aggregate_results()
        
        # Print and save results
        self.print_and_save_results()
        
    def run_trial(self, scenario):
        """Runs trial of a single scenario over multiple CPU cores.
        n_jobs = max cores to use; ise -1 for all available cores. Use of 
        `delayed` ensures different random  numbers in each run"""
        trial_output = Parallel(n_jobs=-1)(delayed(self.single_run)(scenario, i) 
                for i in range(self.replications))

        return trial_output


    def single_run(self, scenario, i=0):
        """Calls for a single run of a single scenario. Returns patient
        count audit, and qtime dictionary (lists of queuing times by patient
        priority)."""

        model = Model(scenario)
        model.run()
        
        # Put results in a dictionary
        results = {
            'all': model.results_summary_all,
            'by_admitting_unit': model.results_summary_by_admitting_unit
                    }
        return results
