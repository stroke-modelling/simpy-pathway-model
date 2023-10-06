from sim_classes.model import Model

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class Replicator:
    
    """Replicator class. Calls multiple replicates of multiple scenarios.
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
        6. When all sceanrios and trials are run, `run_scenarios` will print and
            save results. 

        
    Attributes
    ----------
    patients_pivot:
        Summary statistics of patients waiting and beds occupied for each
        scenario (summarises across replications)
    queue_times_pivot:
        Summary statistics queue times by patient priority for each scenario 
        (summarises across replications).
    replications:
        Number of replications to perform for each scenario.
    scenarios.
        Dictionary of alternative sceanrios (each will be run with multiple
        replications).
    summary_patients:
        Summary statistics of patients waiting and beds occupied for each
        scenario and replication.
    summary_queue_times:
        Summary statistics of waiting times by patient priority for each
        scenario and replications.

    Methods
    -------
    __init__:
        Constructor method for replicator.
    pivot_results:
        Takes summary results for each sceanrio repreplicationslciation and 
        summarises across replications.
    print_results:
        Print results at end of running all scenarios and replications.
        For simplification prints only median and max of each result.
    run_scenarios:
        Calls for replications of each scenario, calls for summarisation, 
        displaying of results, and saving of results.
    run_trial:
        Runs trial of a single scenario over multiple CPU cores
    save_results:
        Save summary results to csv files in output folder
    single_run:
        Calls for a single run of a single scenario

    """



    def __init__(self, scenarios, replications):
        """Constructor method for replicator."""
        
        self.replications = replications
        self.scenarios = scenarios
        
        # Set up DataFrames for all trials results
        self.summary_patients = pd.DataFrame()
        self.summary_queue_times = pd.DataFrame()


    def collate_trial_results(self, name, results):

        for run in range(self.replications):
            # patients summary
            result_item = results[run]['patients']
            result_item['run'] = run
            result_item['name'] = name
            self.summary_patients = self.summary_patients.append(result_item)
            
            # Queue times summary
            result_item = results[run]['queue_times']
            result_item['run'] = run
            result_item['name'] = name
            self.summary_queue_times = \
                self.summary_queue_times.append(result_item)
        
        
    def pivot_results(self):  
        """"""

        # Define functions to allow percentiles in a Pandas Pivot Table
        def percent_5(g):
            return np.percentile(g, 5)

        def percent_95(g):
            return np.percentile(g, 75)

        # Patient counts summary (numbers in beds or wating)
        # ==================================================

        df = self.summary_patients.copy()
        
        # Ensure all numeric data is is a form for aggregation
        df[['min', 'median', 'max']] = \
            df[['min', 'median', 'max']].astype('float')
        
        # Get result type to use for pivot (summarise by this)
        df['result_type'] = df.index
         
        self.patients_pivot = df.pivot_table(
            # Pivot by result type and sceanrio name
            index = ['result_type', 'name'],
            # Summarise min, median and max from each scenario replication
            values = ['min','median','max'],
            # Calculate min, mean, median and max across replications
            aggfunc = [percent_5, np.median, percent_95],
            # Do not include row/column totals
            margins=False)
        
        # Re-order pivot tables column 
        new_order = [(header_1, header_2) for header_1 in 
                     ['percent_5', 'median', 'percent_95'] for header_2 in
                     ['min', 'median', 'max']]
        self.patients_pivot = self.patients_pivot[new_order]
        

        
        # Queue times summary
        # ===================

        # See comments above on patient counts summary for details of steps.

        df = self.summary_queue_times.copy()
        
        df[['min', 'median', 'max']] = \
            df[['min', 'median', 'max']].astype('float')

        df['result_type'] = df.index
         
        self.queue_times_pivot = df.pivot_table(
            index = ['result_type', 'name'],
            values = ['min','median','max'],
            aggfunc = [percent_5, np.median, percent_95],
            margins=False)
        
        # Re-order pivot tables column 
        new_order = [(header_1, header_2) for header_1 in 
                     ['percent_5', 'median', 'percent_95'] for header_2 in
                     ['min', 'median', 'max']]
        self.queue_times_pivot = self.queue_times_pivot[new_order]
        
    
    def print_results(self):
        """Print results at end of running all scenarios and replications.
        For simplification prints only median and max of each result."""
                
        print('\nPatient results')
        print('----------------')
        print('\npercent_5, median, and percent_95 are spread between trials')
        print('min, median and max refer to patient counts in each run\n')
        print(self.patients_pivot)
        print('\n\n')
        print('Queuing times')
        print('-------------')
        print('\npercent_5, median, and percent_95 are spread between trials')
        print('min, median and max refer to waiting times in each run\n')
        print(self.queue_times_pivot)
        

    def run_scenarios(self):
        """Calls for replications of each scenario, calls for summarisation, 
        displaying of results, and saving of results."""
        
        # Run all scenarios
        scenario_count = len(self.scenarios)
        counter = 0
        for name, scenario in self.scenarios.items():
            counter += 1
            print(f'\r>> Running scenario {counter} of {scenario_count}', end='')
            # Call for all replications of a single scenario to be run
            scenario_output = self.run_trial(scenario)
            # Collate trial results from scenario into single DataFrame 
            self.collate_trial_results(name, scenario_output)
        
        # Clear displayed progress output
        clear_line = '\r' + " " * 79
        print(clear_line, end = '')
        
        # Pivot results (Get summary results for all sceanrios)
        self.pivot_results()
        
        # Print results
        self.print_results()
        
        # save results
        self.save_results()
        
        
    def run_trial(self, scenario):
        """Runs trial of a single scenario over multiple CPU cores.
        n_jobs = max cores to use; ise -1 for all available cores. Use of 
        `delayed` ensures different random  numbers in each run"""
        trial_output = Parallel(n_jobs=-1)(delayed(self.single_run)(scenario, i) 
                for i in range(self.replications))
        
        return trial_output
    
    
    def save_results(self):
        """Save summary results to csv files in output folder"""
        
        self.patients_pivot.to_csv('./output/patients.csv')
        self.queue_times_pivot.to_csv('./output/queue_times.csv')


    def single_run(self, scenario, i=0):
            """Calls for a single run of a single scenario. Returns patient
            count audit, and qtime dictionary (lists of queuing times by patient
            priority)."""

            print(f'{i}, ', end='' )
            model = Model(scenario)
            model.run()
            
            # Put results in a dictionary
            results = {
                'patients': model.results_audit,
                'queue_times': model.results_qtime
                       }
            
            return results
        
        
    
    
