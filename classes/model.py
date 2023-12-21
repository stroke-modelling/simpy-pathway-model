"""
Model class for running a simulation of the stroke pathway.
"""
import numpy as np
import pandas as pd
import simpy

from classes.patient import Patient
from classes.pathway import Pathway
# For type hinting:
from classes.scenario import Scenario


class Model(object):
    """
    The main model object.

    class Model():

    Attributes
    ----------
    env:
        SimPy environment

    scenario:
        Scenario class. Imported global variables (object).

    pathway:
        Pathway class. Processes the pathway of each patient.

    Methods
    -------
    __init__:
        Constructor class for model

    end_run_routine:
        Summarise results of the completed simulation.

    generate_patient_arrival:
        SimPy generator to generate a patient.

    run:
        Main model running method.
    """

    def __init__(self, scenario: type[Scenario]):
        """
        Constructor class for model
        """

        # Scenario
        self.scenario = scenario

        # Set up SimPy environment
        self.env = simpy.Environment()

        # Set up pathway
        self.pathway = Pathway(self.env, self.scenario)

    def end_run_routine(self):
        """
        End run routine. Summarise results of the completed simulation.

        Find the mean and standard deviations of each time attribute
        in the pathway, both across all teams and for each team
        individually. The results are stored in dataframes.

        Stores
        ------

        results_all:
            pd.DataFrame. Full results for all patients.

        results_summary_all:
            pd.DataFrame. Summarised (mean and std) results for all
            patients.

        results_summary_by_admitting_unit:
            pd.DataFrame. Summarised (mean and std) results for each
            stroke team individually.
        """

        # Record which columns are used for times.
        # self.pathway.completed_patients is a list of dictionaries
        # with shared keys.
        # Get all names in the completed patient dictionaries:
        completed_patients_keys = self.pathway.completed_patients[0].keys()
        # Keep only those that begin with "time":
        time_cols = [x for x in completed_patients_keys if x[0:4] == 'time']
        # Remove the onset time key:
        time_cols.remove('time_onset')

        # Convert results into DataFrames
        # self.pathway.completed_patients is a list of dictionaries
        # with shared keys so can be converted to DataFrame:
        self.results_all = pd.DataFrame(self.pathway.completed_patients)
        # Take only the columns relating to time
        # and take only their means and standard deviations:
        self.results_summary_all = (
            self.results_all[time_cols].agg(['mean', 'std']))
        # Rename the index column:
        self.results_summary_all.index.name = 'statistic'

        # Group the results by first unit.
        # Group by unit, then take only the columns relating to time,
        # then take only their means and standard deviations.
        self.results_summary_by_admitting_unit = self.results_all.groupby(
            by='closest_ivt_unit')[time_cols].agg(['mean', 'std'])

    def generate_patient_arrival(self):
        """
        SimPy generator to generate a patient.

        Patients are generated in a continuous loop with a SimPy
        time-out between arrivals. The patient is passed to the
        pathway (pathway.process_patient) which directs the patient
        journey, and removes the patient at the end of the journey.

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
        """
        Model run: Initialise processes needed at model start,
        start model running, and call end_run_routine.
        Note: All SimPy processes must be called with `env.process`
        in addition to the process function/method name
        """

        # Initialise processes that will run on model run
        self.env.process(self.generate_patient_arrival())

        # Run
        self.env.run(until=self.scenario.run_duration)

        # Call end run routine to summarise and record model results
        self.end_run_routine()
