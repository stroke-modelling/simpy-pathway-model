class Patient():
    """
    Attributes
    ----------
    los:
        length of stay in hospital
    priority:
        priority for accessing bed (loer number = higher priority)
    time_enter_queue:
        time patient arrives and joins queue for bed
    time_leave_queue:
        time patient leaves queue and enters hospital bed

    Methods
    -------

    __init__:
        Constructor class for patient
    """

    def __init__(self, los=1, priority=2):
        self.los = los
        self.priority = priority
        self.time_enter_queue = 0
        self.time_leave_queue = 0

        