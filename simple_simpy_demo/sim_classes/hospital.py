import pandas as pd
import queue
from simpy import PriorityResource


class Hospital():
    """
    Attributes
    ----------
    _env:
        Reference to simpy environment
    _params:
        Reference to model prarameters
    beds:
        SimPy resource of beds
    count_patient_waiting_by_priority:
        dictionary of patients waiting, key=priority(1-3)
    patients_waiting:
        List of patients waiting to be allocated to a bed
    patients_in_bed:
        List of patients allocated to beds
    queue_time_by_priority:
        dictionary of queuing times by priority, key=priority(1-3)

    Methods
    -------
    __init__:
        Constrcutor class for hospital

    allocate_bed:
        Simpy process. Manages queues for beds, and allocates bed by patient 
        priority.

    """

    def __init__(self, env, params, number_of_beds):
        """Constructor method for hospital"""

        self._env = env
        self._params = params
        self.beds = PriorityResource(env, capacity=number_of_beds)
        self.count_patient_waiting_by_priority = {1: 0, 2: 0, 3: 0}
        self.patients_waiting = []
        self.patients_in_bed = []
        self.queue_time_by_priority = {1: [], 2: [], 3: []}

        # Set up DataFrame for daily audit
        columns = ['in_bed','waiting_p1', 'waiting_p2', 'waiting_p3']
        self.audit = pd.DataFrame(columns=columns)

    
    def allocate_bed(self, patient):
        """SimPy process. Add patient to queue waiting for bed. Allocate beds by
        patient priority. Trach queueing times"""

        # Put patient in list of wating patients.
        patient.time_enter_queue = self._env.now
        self.patients_waiting.append(patient)
        self.count_patient_waiting_by_priority[patient.priority] += 1
        
        # Create request for bed resources (by patient priority)
        req = self.beds.request(priority=patient.priority)
        
        # Request bed (and wait at this stage until bed is available)
        yield req
        
        # Resources now available. Get queuing time
        patient.time_leave_queue = self._env.now
        queue_time = patient.time_leave_queue - patient.time_enter_queue
        self.count_patient_waiting_by_priority[patient.priority] -= 1
        
        # Add to queuing times if passed model warm up period
        if self._env.now >= self._params.warm_up:
            self.queue_time_by_priority[patient.priority].append(queue_time)
        
        # Remove from patients_waiting, and add to patients_in_bed
        self.patients_waiting.remove(patient)
        self.patients_in_bed.append(patient)
        
        # Wait patient length of stay using environment timeout
        yield self._env.timeout(patient.los) 
        
        # Patient stay in hospital has ended. Remove patient
        self.patients_in_bed.remove(patient)
        self.beds.release(req)
        del patient


    def perform_audit(self):
        """SimPy process. Count numbers in hospital and waiting for a bed"""

        # delay before first_audit
        yield self._env.timeout(self._params.warm_up)

        # Coontinual audit loop
        while True:
            # Set up dictionary for results
            results = dict()
            # Get numebrs in beds and waiting (by priority)
            results['in_bed'] = len(self.patients_in_bed)
            results['waiting_p1'] = self.count_patient_waiting_by_priority[1]
            results['waiting_p2'] = self.count_patient_waiting_by_priority[2]
            results['waiting_p3'] = self.count_patient_waiting_by_priority[3]
            # Add results to audit DataFrame
            self.audit = self.audit.append(results,ignore_index=True)
            # 1 day delay before continuing with while loop
            yield self._env.timeout(1)
        










