import numpy as np
import pandas as pd
import simpy
import time

# Imports from the stroke_outcome package:
from stroke_outcome.discrete_outcome import Discrete_outcome
import stroke_outcome.outcome_utilities as outcome_utilities

from classes.parameters import Scenario
from classes.patient import Patient
from classes.pathway import Pathway


class Model(object):
    """
    The main model object.

    class Model():

    Attributes
    ----------
    env:
        SimPy environment

    scenario:
        Imported global variables (object)

    Methods
    -------
    __init__:
        Constructor class for model

    """

    def __init__(self, scenario):
        """Constructor class for model"""

        # Scenario
        self.scenario = scenario

        # Set up SimPy environment
        self.env = simpy.Environment()

        # Set up pathway
        self.pathway = Pathway(self.env, scenario)

    def end_run_routine(self):
        '''
        End run routine
        '''

        # Record which columns are used for times
        completed_patients_keys = self.pathway.completed_patients[0].keys()
        time_cols = [x for x in completed_patients_keys if x[0:4] == 'time']
        time_cols.remove('time_onset')

        # Convert results into DataFrames
        self.results_all = pd.DataFrame(self.pathway.completed_patients)
        self.results_summary_all = self.results_all[time_cols].agg(['mean', 'std'])
        self.results_summary_all.index.name = 'statistic'

        self.results_summary_by_admitting_unit = self.results_all.groupby(
            by='closest_ivt_unit')[time_cols].agg(['mean', 'std'])

    def generate_patient_arrival(self):
        """
        SimPy generator to generate a patient. Patients are generated in a continuous
        loop with a SimPy time-out between arrivals. The patient is passed to the 
        pathway (pathway.process_patient) which directs the patient journey, and removes
        the patient at the end of the journey.

        Returns:
        --------
        none 

        """

        # Continuous loop of patient arrivals
        arrival_count = 0
        while True:
            arrival_count += 1
            # Get patient object
            patient = Patient(self.scenario, arrival_count)
            # Pass patient to pathway
            self.env.process(self.pathway.process_patient(patient))
            # Sample time to next admission from exponential distribution
            time_to_next = np.random.exponential(
                self.scenario.inter_arrival_time)
            # SimPy delay to next arrival (using environment timeout)
            yield self.env.timeout(time_to_next)

    def run(self):
        """Model run: Initialise processes needed at model start, start model 
        running, and call end_run_routine.
        Note: All SimPy processes must be called with `env.process` in addition
        to the process function/method name"""

        # Initialise processes that will run on model run
        self.env.process(self.generate_patient_arrival())

        # Run
        self.env.run(until=self.scenario.run_duration)

        # Call end run routine to summarise and record model results
        self.end_run_routine()
