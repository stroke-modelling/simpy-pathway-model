import numpy as np

class Scenario(object):
    """
    Model scenario parameters.
    
    Attributes
    ----------
    hospital_beds:
        Total number of beds in hospital.
    interarrival_time:
        Mean ime (days) between parient arrival.
    patient_los:
        Mean patient length of stay (days). 
        Exponential distibution will be applied.
    run_duration:
        Simulation run time (including warm-up).
    warm_up:
        Simulation run time before audit starts.

    Methods:
    __innit__:
        Constructor method.

    Note: Using Python DataClass (Python 3.7 onwards) or using getters/setters
    can help prevent parameters being accidently changed elsewhere in the
    programme. 
    """
    
    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""

        self.interarrival_time = 0.05
        self.hospital_beds = 200
        self.patient_los = 10
        self.run_duration = 100
        self.warm_up = 50

        # Overwrite default values (can take named arguments or a dictionary)
        
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        
        for key in kwargs:
            setattr(self, key, kwargs[key])
            