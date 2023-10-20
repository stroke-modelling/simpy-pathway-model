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
        1. `run_scenarios` is called from outside method.
        2. `run_scenarios` calls `run_trial` for each scenario.
        3. `run_trial` calls `single_run` for each replicate run. This is 
            performed across multiple CPU cores. This is repeated for each
            scenario.
        4. `run_scenarios` then calls `aggregate_results` to aggregate results
            from across trials.
        5. `run_scenarios` then saves and prints aggregate results. For printing,
            only the median of the trials is shown. When saved, the 5%, 50%, and
            95% percentiles across trials are saved.
        """

    def __init__(self, scenarios, replications):
        """Constructor method for replicator."""
        
        self.replications = replications
        self.scenarios = scenarios
        
        # Set up dictionary for concatenated results
        self.results = dict()


    def aggregate_results(self):
        """
        Results across trials are aggregated. The 5%, 50% (median), and 95%
        percentiles are calculated for each result table for each scenario, and
        saved in a dictionary `aggregated_results`.
        """

        # Define functions to allow percentiles in a Pandas Pivot Table
        def percent_5(g):
            """Returns 5th percentile of input"""
            return np.percentile(g, 5)

        def percent_95(g):
            """Returns 95th percentile of input"""
            return np.percentile(g, 75)

        # Store aggregated results in a dictionary
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
        """
        Aggregated results (across multiple trial runs) are saved in full. 
        Printed results are limited to median results from trials. When run
        from a Jupyter notebook, printing  uses `display` rather than `print`
        for nicer formatting of output.

        """

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
                df.to_csv(f'./output/{scenario}_{result_name}.csv')

                # Display median results only (and remove 'median' as column name)
                display_cols = []
                for col in df.columns.values.tolist():
                   if col[-1] == 'median':
                       display_cols.append(col)

                df = df[display_cols]
                df.columns = df.columns.droplevel(-1) # Drop 'median' label

                print ()
                print (f'Scenario: {scenario}, Result: {result_name}')
                # Use display from within Jupyter notebook
                try:
                    display (df)
                except:
                    print (df)


    def run_scenarios(self):
        """
        Calls for replications of each scenario, calls for summarisation, 
        displaying of results, and saving of results.
        """
        
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
                # Concatenate trial results
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
        """Calls for a single run of a single scenario."""

        model = Model(scenario)
        model.run()
        
        # Put model results in a dictionary for returning to main code.
        results = {
            'all': model.results_summary_all,
            'by_admitting_unit': model.results_summary_by_admitting_unit
                    }

        return results
