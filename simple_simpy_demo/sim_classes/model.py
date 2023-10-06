import numpy as np
import pandas as pd
import simpy

from sim_classes.patient import Patient
from sim_classes.hospital import Hospital


class Model():
    """
    A simple hospital model. The Model sets up a Hospital object.

    Attributes
    ----------
    _env:
        SimPy environment
    _parmas:
        Reference to model parameters
    hospital:
        Hospital object

    Methods
    -------

    __init__:
        Constructor class for model
    end_run_routine:
        Summarise patient counts and queue times
    generate_patient_arrival:
        SimPy process. Generates new patients in mode. 
    run:
        Model run: Initialise processes needed at model start, start model 
        running, and call end_run_routine


    """

    def __init__(self, params):
        """Constrcutor class for model"""

        self._env = simpy.Environment()
        self._params = params
        self.hospital = Hospital(env=self._env, params=self._params,
                                 number_of_beds=self._params.hospital_beds)
    
    def end_run_routine(self):
        """Summarise patient counts and queue times"""
        
        # Summarise daily audit
        self.results_audit = pd.DataFrame()
        self.results_audit['min'] = self.hospital.audit.min(axis=0)
        self.results_audit['median'] = self.hospital.audit.median(axis=0)
        self.results_audit['max'] = self.hospital.audit.max(axis=0)
        
        # Set up DataFrame for queuing time stats
        self.results_qtime = pd.DataFrame(index=['P1', 'P2','P3'], 
                                          columns=['min', 'median', 'max'])
        
        # Get stats for each priority (loop through priorities)
        for priority in range(1,4):
            # key for summary DataFrame is P1, P2, P3...
            key = 'P' + str(priority)
            # Get minumim, median and max time from list of queue times
            self.results_qtime.loc[key]['min'] = \
                np.min(self.hospital.queue_time_by_priority[priority])
            self.results_qtime.loc[key]['median'] = \
                np.median(self.hospital.queue_time_by_priority[priority])
            self.results_qtime.loc[key]['max'] = \
                np.max(self.hospital.queue_time_by_priority[priority])


    def generate_patient_arrival(self):
        """SimPy process. Generate patients. Assign priority and length of stay. 
        Pass patient to hospital bed allocation"""
        
        # Continuous loop of patient arrivals
        while True:
            # Sample patient priority from uniform distribution
            priority = np.random.randint(1, 4)
            # Sample patient length of stay from exponential distribution
            los = np.random.exponential(self._params.patient_los)
            # Create patient object
            patient = Patient(los=los, priority=priority)
            # Pass patient to hospital bed allocation process
            self._env.process(self.hospital.allocate_bed(patient))
            # Sample time to next admission from exponential distribution
            time_to_next = np.random.exponential(self._params.interarrival_time)
            # SimPy delay to next arrival (using environment timeout)
            yield self._env.timeout(time_to_next)
            # Return to top of while loop


    def run(self):
        """Model run: Initialise processes needed at model start, start model 
        running, and call end_run_routine.
        Note: All SimPy processes must be called with `env.process` in addition
        to the process function/method name"""

        # Initialise processes that will run on model run. 
        self._env.process(self.generate_patient_arrival())
        self._env.process(self.hospital.perform_audit())

        # Run
        self._env.run(until=self._params.run_duration)
        
        # End of run
        self.end_run_routine()
        

