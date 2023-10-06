import random


class Pathway(object):
    """
    Stroke pathway processes
    """

    def __init__(self, env, globvars):
        """Constructor class"""

        self.env = env
        self.globvars = globvars

    def process_patient(self, patient):
        """
        Manages the pathway process steps for each patient
        """

        # Onset
        patient.times["onset"] = self.env.now

        # Call ambulance
        yield self.env.process(self.call_ambulance(patient))

        # Ambulance response
        yield self.env.process(self.ambulance_response(patient))
        patient.times["ambulance_arrives"] = self.env.now

        # End of pathway
        patient.times["end"] = self.env.now

        # Delete patient object
        del patient

    def ambulance_response(self, patient):
        """
        Time from calling ambulance to arrival
        """
        min_duration = self.globvars.process_time_ambulance_response[0]
        max_duration = self.globvars.process_time_ambulance_response[1]
        duration = random.uniform(min_duration, max_duration)
        yield self.env.timeout(duration)
        patient.times["ambulance_arrives"] = self.env.now

    def call_ambulance(self, patient):
        """
        Time from onset to calling for ambulance
        """
        min_duration = self.globvars.process_time_call_ambulance[0]
        max_duration = self.globvars.process_time_call_ambulance[1]
        duration = random.uniform(min_duration, max_duration)
        yield self.env.timeout(duration)
        patient.times["ambulance_called"] = self.env.now
