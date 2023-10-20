import numpy as np
import random


class Pathway(object):
    """
    Stroke pathway processes.

    ...

    Attributes
    ----------
    complete_patients: list
        List of dictionaries. Each dictionary records info for each completed
        patient journey.

    env: SimPy environment object
        SimPy environment passed to the pathway

    globvars: globvars object
        Global variables for passed to the pathway


    Methods
    -------
    ambulance_response:
        Time from calling ambulance to arrival of ambulance

    process_patient:
        Manages the pathway process steps for each patient.
    """

    def __init__(self, env, scenario):
        """Constructor class"""

        self.env = env
        self.scenario = scenario
        self.completed_patients = []

    def process_patient(self, patient):
        """
        Manages the pathway process steps for each patient
        """

        # Onset
        patient.time_onset = np.round(self.env.now, 1)

        # Call ambulance
        yield self.env.process(self.call_ambulance(patient))

        # Delay before ambulance call
        yield self.env.process(self.ambulance_response(patient))

        # Ambulance response
        yield self.env.process(self.ambulance_response(patient))

        # Record patient info and delete patient
        self.completed_patients.append(patient.__dict__)
        del patient

    def ambulance_response(self, patient):
        """
        Time from calling ambulance to arrival of ambulance
        """
        min_duration = self.scenario.process_time_ambulance_response[0]
        max_duration = self.scenario.process_time_ambulance_response[1]
        duration = random.uniform(min_duration, max_duration)
        yield self.env.timeout(duration)
        patient.time_ambulance_arrives = np.round(
            self.env.now - patient.time_onset, 1)

    def call_ambulance(self, patient):
        """
        Time from onset to calling for ambulance
        """
        min_duration = self.scenario.process_time_call_ambulance[0]
        max_duration = self.scenario.process_time_call_ambulance[1]
        duration = random.uniform(min_duration, max_duration)
        yield self.env.timeout(duration)
        patient.time_ambulance_called = np.round(
            self.env.now - patient.time_onset, 1)
