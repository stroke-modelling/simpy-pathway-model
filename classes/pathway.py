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

    def process_patient(self, patient, scenario):
        """
        Manages the pathway process steps for each patient
        """

        # Onset
        patient.time_onset = np.round(self.env.now, 1)

        # Call ambulance
        yield self.env.process(self.call_ambulance(patient))

        # Delay before ambulance call
        # ??????????????????????????????????????????????????????????????????????????
        yield self.env.process(self.ambulance_response(patient))

        # Ambulance response
        yield self.env.process(self.ambulance_response(patient))

        # Choose unit
        self.choose_unit(patient)

        # Travel to unit
        yield self.env.process(self.go_to_unit(patient))

        # Go to scanner
        yield self.env.process(self.go_to_scanner(patient, scenario))

        # Choose whether patient thrombolysed.
        self.choose_whether_thrombolysis(patient)

        # Go to thrombolysis
        yield self.env.process(self.go_to_thrombolysis(patient, scenario))

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

    def choose_unit(self, patient):
        """
        Choose whether to travel to nearest IVT or nearest MT unit.

        PLACEHOLDER - this function is basic for now.
        Perhaps later we'll use other patient attributes to make
        a realistic choice.
        """
        # PLACEHOLDER selection:
        c = 0.25  # Chance of the MT unit being picked
        ivt_chosen = (np.random.binomial(1, c) == 0)

        # Pick out either the IVT or MT unit info:
        if ivt_chosen:
            unit = patient.closest_ivt_unit
            time_to_unit = patient.closest_ivt_duration
        else:
            unit = patient.closest_mt_unit
            time_to_unit = patient.closest_mt_duration
        # Store in self:
        patient.first_unit = unit
        patient.duration_ambulance_first_unit = time_to_unit

    def go_to_unit(self, patient):
        """
        Time from onset to arrival at unit.
        """
        duration = patient.duration_ambulance_first_unit
        yield self.env.timeout(duration)
        patient.time_unit_arrival = np.round(
            self.env.now - patient.time_onset, 1)

    def go_to_scanner(self, patient, scenario):
        """
        Time from arrival at hospital to scan.

        Uses lognorm distribution from hospital performance.
        """
        mu = (
            scenario.hospital_performance[
                'lognorm_mu_arrival_scan_arrival_mins_ivt'
                ].loc[patient.first_unit]
            )
        sigma = (
            scenario.hospital_performance[
                'lognorm_sigma_arrival_scan_arrival_mins_ivt'
                ].loc[patient.first_unit]
            )

        duration = np.random.lognormal(mu, sigma)
        yield self.env.timeout(duration)
        patient.time_scan = np.round(
            self.env.now - patient.time_onset, 1)

    def choose_whether_thrombolysis(self, patient):
        """
        Choose whether this patient receives thrombolysis.

        PLACEHOLDER - this function is basic for now.
        Perhaps later we'll use other patient attributes to make
        a realistic choice.
        """
        # PLACEHOLDER selection:
        c = 0.2  # Chance of thrombolysis
        b = np.random.binomial(1, c)

        # Store in self:
        patient.thrombolysis = (b == 1)  # Convert to bool

    def go_to_thrombolysis(self, patient, scenario):
        """
        Time from scan to thrombolysis.

        Uses lognorm distribution from hospital performance.
        """
        if patient.thrombolysis:
            mu = (
                scenario.hospital_performance[
                    'lognorm_mu_scan_needle_mins_ivt'
                    ].loc[patient.first_unit]
                )
            sigma = (
                scenario.hospital_performance[
                    'lognorm_sigma_scan_needle_mins_ivt'
                    ].loc[patient.first_unit]
                )

            duration = np.random.lognormal(mu, sigma)
            yield self.env.timeout(duration)
            patient.time_thrombolysis = np.round(
                self.env.now - patient.time_onset, 1)
        else:
            # Leave default value when thrombolysis not given.
            pass
