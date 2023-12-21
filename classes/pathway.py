"""
Pathway class for processing patients with stroke.
"""
import numpy as np
import random


class Pathway(object):
    """
    Stroke pathway processes.

    The pathway is set up using a simpy Environment and
    an instance of the Scenario class (for constants and
    model types). The pathway is used to process patient
    details in the Patient class. Each method in this
    pathway acts on one patient instance at a time.

    class Pathway():

    Attributes
    ----------

    complete_patients: list
        List of dictionaries. Each dictionary records info for each
        completed patient journey.

    env: SimPy environment object
        SimPy environment passed to the pathway.

    scenario: Scenario object
        Class containing global variables for the pathway.


    Methods
    -------

    ambulance_on_scene:
        Wait from arrival of ambulance to departure.

    ambulance_response:
        Wait from calling ambulance to arrival of ambulance.

    call_ambulance:
        Wait from onset to patient calling ambulance.

    choose_admitting_unit:
        Choose which unit to travel to first.

    choose_whether_thrombectomy:
        Generate decision about thrombectomy.

    choose_whether_thrombolysis:
        Generate decision about thrombolysis.

    go_to_admitting_unit:
        Wait for duration of travel to admitting unit.

    go_to_thrombectomy:
        Wait from arrival at MT unit to start of thrombectomy.

    go_to_thrombolysis:
        Wait from scan to start of thrombolysis.

    go_to_transfer_unit:
        Wait for duration of travel to transfer unit.

    process_patient:
        Manages the pathway process steps for each patient.
    """

    def __init__(
            self,
            env: type[Environment],
            scenario: type[Scenario]
            ):
        """Constructor class"""

        self.env = env
        self.scenario = scenario
        self.completed_patients = []

    def process_patient(self, patient: type[Patient]):
        """
        Manages the pathway process steps for each patient.

        Main method for running the other methods in this class.
        """

        # Onset time since start of simulation.
        patient.time_onset = np.round(self.env.now, 1)

        # Call ambulance
        yield self.env.process(self.call_ambulance(patient))

        # Ambulance travels to patient
        yield self.env.process(self.ambulance_response(patient))

        # Ambulance on scene for some duration
        yield self.env.process(self.ambulance_on_scene(patient))

        # Choose unit
        self.choose_admitting_unit(patient)

        # Travel to unit
        yield self.env.process(self.go_to_admitting_unit(patient))

        # Choose which treatments will be given.
        self.choose_whether_thrombolysis(patient)
        self.choose_whether_thrombectomy(patient)

        # Go to thrombolysis if chosen.
        yield self.env.process(self.go_to_thrombolysis(patient))
        # Travel to transfer unit if necessary.
        yield self.env.process(self.go_to_transfer_unit(patient))
        # Go to thrombectomy if chosen.
        yield self.env.process(self.go_to_thrombectomy(patient))

        # The end.
        # Record patient info and delete patient
        self.completed_patients.append(patient.__dict__)
        del patient

    def call_ambulance(self, patient: type[Patient]):
        """
        Time from onset to calling for ambulance.

        For the duration from the end of the previous step,
        choose at random a time between and including
        the two time limits defined in the scenario.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.time_ambulance_called:
            float. Time from onset to calling ambulance.
        """
        # Pick out time limits:
        min_duration = self.scenario.process_time_call_ambulance[0]
        max_duration = self.scenario.process_time_call_ambulance[1]
        # Randomly pick a time in this range:
        duration = random.uniform(min_duration, max_duration)
        # Let this time pass in the simulation.
        yield self.env.timeout(duration)
        # How long has it been since the onset time?
        # Store this length of time with this patient's details.
        patient.time_ambulance_called = np.round(
            self.env.now - patient.time_onset, 1)

    def ambulance_response(self, patient: type[Patient]):
        """
        Time from calling ambulance to arrival of ambulance.

        For the duration from the end of the previous step,
        choose at random a time between and including
        the two time limits defined in the scenario.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.time_ambulance_arrival:
            float. Time from onset to ambulance arrival at onset location.
        """
        # Pick out time limits:
        min_duration = self.scenario.process_time_ambulance_response[0]
        max_duration = self.scenario.process_time_ambulance_response[1]
        # Randomly pick a time in this range:
        duration = random.uniform(min_duration, max_duration)
        # Let this time pass in the simulation.
        yield self.env.timeout(duration)
        # How long has it been since the onset time?
        # Store this length of time with this patient's details.
        patient.time_ambulance_arrival = np.round(
            self.env.now - patient.time_onset, 1)

    def ambulance_on_scene(self, patient: type[Patient]):
        """
        Duration of ambulance initially with patient.

        For the duration from the end of the previous step,
        choose at random a time between and including
        the two time limits defined in the scenario.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.time_ambulance_leaves_scene:
            float. Time from onset to calling ambulance.
        """
        # Pick out time limits:
        min_duration = self.scenario.process_ambulance_on_scene_duration[0]
        max_duration = self.scenario.process_ambulance_on_scene_duration[1]
        # Randomly pick a time in this range:
        duration = random.uniform(min_duration, max_duration)
        # Let this time pass in the simulation.
        yield self.env.timeout(duration)
        # How long has it been since the onset time?
        # Store this length of time with this patient's details.
        patient.time_ambulance_leaves_scene = np.round(
            self.env.now - patient.time_onset, 1)

    def choose_admitting_unit(self, patient: type[Patient]):
        """
        Choose which unit to travel to first.

        Choose whether to travel to nearest IVT or nearest MT unit.
        In the drip-and-ship model, the nearest IVT unit is picked.
        Other models are not yet implemented.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.admitting_unit:
            str. Name of the admitting unit the patient goes to.

        patient.admitting_unit_travel_duration:
            float. Time to travel to the admitting unit.
        """
        if self.scenario.destination_decision_type == 0:
            # Drip and ship model.
            ivt_unit_chosen = True
        else:
            # TO DO - implement mothership or other ways to pick
            # and choose in some known ratio.
            # PLACEHOLDER selection:
            c = 0.25  # Chance of the MT unit being picked
            ivt_unit_chosen = (np.random.binomial(1, c) == 0)

        # Pick out the chosen unit info:
        if ivt_unit_chosen:
            # Use IVT unit details.
            unit = patient.closest_ivt_unit
            time_to_unit = patient.closest_ivt_travel_duration
        else:
            # Use MT unit details.
            unit = patient.closest_mt_unit
            time_to_unit = patient.closest_mt_travel_duration
        # Store with this patient's details:
        patient.admitting_unit = unit
        patient.admitting_unit_travel_duration = time_to_unit

    def go_to_admitting_unit(self, patient: type[Patient]):
        """
        Time from onset to arrival at unit.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.time_admitting_unit_arrival:
            float. Time from onset to arrival at admitting unit.
        """
        # How long does it take to travel to admitting unit?
        duration = patient.admitting_unit_travel_duration
        # Let this time pass in the simulation.
        yield self.env.timeout(duration)
        # How long has it been since this patient's stroke onset?
        # Store this length of time with this patient's details.
        patient.time_admitting_unit_arrival = np.round(
            self.env.now - patient.time_onset, 1)

    def choose_whether_thrombolysis(self, patient: type[Patient]):
        """
        Choose whether this patient receives thrombolysis.

        If the patient does not have ischaemic stroke, then
        the chance of thrombolysis is zero. Otherwise the
        probability of thrombolysis is taken from the scenario.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.thrombolysis:
            bool. True if the patient is chosen for thrombolysis,
            otherwise False.
        """
        if patient.stroke_type in [1, 2]:
            # This patient has ischaemic stroke
            # (their stroke type code is for nLVO or LVO).
            # Flat chance of thrombolysis regardless of pathway
            # timings and other patient characteristics.
            # Chance:
            c = self.scenario.probability_ivt
            # Choose whether they receive IVT:
            b = np.random.binomial(1, c)
        else:
            # This patient does not have ischaemic stroke.
            # They must not receive thrombolysis.
            b = 0

        # Store the selection with this patient's details:
        patient.thrombolysis = (b == 1)  # Convert to bool

    def choose_whether_thrombectomy(self, patient: type[Patient]):
        """
        Choose whether this patient receives thrombolysis.

        If the patient does not have ischaemic stroke, then
        the chance of thrombectomy is zero. Otherwise the
        probability of thrombectomy is taken from the scenario.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.thrombectomy:
            bool. True if the patient is chosen for thrombectomy,
            otherwise False.
        """
        if patient.stroke_type == 2:
            # This patient has an LVO.
            # Flat chance of thrombectomy regardless of pathway
            # timings and other patient characteristics.
            # Chance:
            c = self.scenario.probability_mt
            # Choose whether they receive MT:
            b = np.random.binomial(1, c)
        else:
            # This patient does not have an LVO.
            # They must not receive thrombectomy.
            b = 0

        # Store the selection with this patient's details:
        patient.thrombectomy = (b == 1)  # Convert to bool

    def go_to_thrombolysis(self, patient: type[Patient]):
        """
        Time from scan to thrombolysis.

        If thrombolysis is not chosen, then nothing happens.
        If thrombolysis is chosen, find the time that it happens.
        For the duration from the end of the previous step,
        choose at random a time between and including
        the two time limits defined in the scenario.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.time_needle:
            float. Time from onset to needle. If thrombolysis was
            not given, then this is left as the initialised value
            in the Patient object.
        """
        if not patient.thrombolysis:
            # Nothing happens when thrombolysis is not given.
            pass
        else:
            # Find the time before thrombolysis starts.
            # Pick out time limits:
            min_duration = self.scenario.process_time_arrival_to_needle[0]
            max_duration = self.scenario.process_time_arrival_to_needle[1]
            # Randomly choose a time in this range:
            duration = random.uniform(min_duration, max_duration)
            # Let this time pass in the simulation.
            yield self.env.timeout(duration)
            # How long has it been since this patient's stroke onset?
            # Store this length of time with this patient's details.
            patient.time_needle = np.round(
                self.env.now - patient.time_onset, 1)

    def go_to_transfer_unit(self, patient: type[Patient]):
        """
        Time from onset to arrival at transfer unit.

        If the patient does not need to be transferred for
        thrombectomy then nothing happens. Otherwise there
        is a delay in the admitting unit and then the travel
        time to the MT unit.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.time_transfer_unit_arrival:
            float. Time from onset to arrival at transfer unit.
            If transfer was not required, then this is left as the
            initialised value in the Patient object.
        """
        if (patient.thrombectomy & patient.mt_transfer_required):
            # If the patient needs MT in another unit,
            # find how long it will take for the transfer and
            # travel time:
            duration = (self.scenario.transfer_time_delay +
                        patient.mt_transfer_travel_duration)
            # Let this time pass in the simulation:
            yield self.env.timeout(duration)
            # How long has it been since this patient's stroke onset?
            # Store this length of time with this patient's details.
            patient.time_transfer_unit_arrival = np.round(
                self.env.now - patient.time_onset, 1)
        else:
            # Nothing happens.
            # This condition is met when:
            #   - the patient needs thrombectomy and is already in
            #     a unit that offers MT, or
            #   - the patient does not need thrombectomy.
            pass

    def go_to_thrombectomy(self, patient: type[Patient]):
        """
        Time from scan to thrombectomy.

        If the patient was not chosen for thrombectomy then
        nothing happens. Otherwise, different time delays
        before the start of thrombectomy are used depending
        on whether MT is given in the admitting unit or a transfer unit.
        The delays are both defined as from arrival in the unit
        providing MT.

        For the duration from the end of the previous step,
        choose at random a time between and including
        the two time limits defined in the scenario.
        Then if the admitting unit is the MT unit, make sure that the
        delay is timed from arrival by subtracting
        any already elapsed time in the admitting unit.

        Inputs
        ------

        patient:
            Patient object.

        Result
        ------

        patient.time_puncture:
            float. Time from onset to puncture. If thrombectomy was
            not given, then this is left as the initialised value
            in the Patient object.
        """
        if not patient.thrombectomy:
            # Nothing happens when thrombectomy not given.
            pass
        else:
            if patient.mt_transfer_required:
                # This patient receives MT in a transfer unit.
                # Another method has already added on the transfer
                # delay and travel time. Here we find just the
                # duration from arrival at transfer unit to puncture.
                # Pick out the time limits:
                min_duration = (
                    self.scenario.process_time_transfer_arrival_to_puncture[0])
                max_duration = (
                    self.scenario.process_time_transfer_arrival_to_puncture[1])
                # Randomly choose a time in this range:
                duration = random.uniform(min_duration, max_duration)
            else:
                # This patient receives MT in their admitting unit.
                # Pick out the time limits:
                min_duration = (
                    self.scenario.process_time_arrival_to_puncture[0])
                max_duration = (
                    self.scenario.process_time_arrival_to_puncture[1])
                # Randomly choose a time in this range:
                duration = random.uniform(min_duration, max_duration)
                # How long has it been since arrival at admitting unit?
                time_in_admitting_unit = (
                    self.env.now - patient.time_admitting_unit_arrival)
                # Subtract this amount from the requested duration
                # (which is the time from arrival at admitting unit to
                # treatment).
                duration = np.max(((duration - time_in_admitting_unit), 0.0))
            # Let the chosen time pass in the simulation.
            yield self.env.timeout(duration)
            # How long has it been since this patient's stroke onset?
            # Store this length of time with this patient's details.
            patient.time_puncture = np.round(
                self.env.now - patient.time_onset, 1)
