import numpy as np
import pandas as pd


class Scenario(object):
    """
    Global variables for model.

    Parameters:
    -----------

    hospitals:
        Info on stroke hospitals. Pandas DataFrame.

    inter_arrival_time:
        Time (minutes) between arrivals. Decimal.

    limit_to_england:
        Limit model to only England admissions. Boolean

    lsoa_names:
        List of LSOA names. List.

    lsoa_relative_frequency:
        Relative frequency of admissions to each LSOA (sums to 1). NumPy array.

    lsoa_ivt_travel_time:
        Travel time (minutes) from LSOA to closest IVT unit. Dictionary.

    lsoa_ivt_unit:
        Closest IVT unit postcode for each LSOA. Dictionary.

    lsoa_mt_travel_time:
        Travel time (minutes) from LSOA to closest MT unit. Dictionary.

    lsoa_mt_unit:
        Closest MT unit postcode for each LSOA. Dictionary.

    mt_transfer_time:
        Time (minutes) for closest IVT to MT transfer. Dictionary.

    mt_transfer_unit:
        Closest MT unit for each IVT unit.  Dictionary.

    process_time_ambulance_response:
        Min/max of time from 999 call to ambulance arrival (tuple of integers)

    run_duration:
        Simulation run time (minutes, including warm-up). Integer.

    total_admissions:
        Total yearly admissions (obtained from LSOA admissions). Float.

    warm_up:
        Simulation run time (minutes) before audit starts.


    Methods:
    --------
    load_data:
        Loads data to be used

    """

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""

        self.limit_to_england = True
        self.run_duration = 365 #Days
        self.warm_up = 50

        # Load data
        self.load_data()

        # Set process times
        self.process_time_ambulance_response = (10, 40)
        self.process_time_call_ambulance = (5, 60)

        # Overwrite default values (can take named arguments or a dictionary)

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Convert run duration to minutes
        self.run_duration *= 1440

    def load_data(self):
        """
        Load required data.

        Stores the following in the Globvars object:

            hospitals
            lsoa_ivt_travel_time
            lsoa_ivt_unit
            lsoa_mt_travel_time
            lsoa_mt_unit
            lsoa_names
            lsoa_relative_frequency
            lsoa_travel_time
            mt_transfer_time
            mt_transfer_unit
            total_admissions
        """

        # Load and parse admissions data
        admissions = pd.read_csv("./data/admissions_2017-2019.csv")
        if self.limit_to_england:
            mask = admissions["England"] == 1
            admissions = admissions[mask]

        self.total_admissions = np.round(admissions["Admissions"].sum(), 0)
        self.lsoa_relative_frequency = np.array(
            admissions["Admissions"] / self.total_admissions
        )
        self.lsoa_names = list(admissions["area"])

        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions

        # Load and parse hospital data
        hospitals = pd.read_csv("./data/stroke_hospitals_2022.csv")
        hospitals["Use"] = hospitals[["Use_IVT", "Use_MT"]].max(axis=1)
        mask = hospitals["Use"] == 1
        hospitals = hospitals[mask]
        if self.limit_to_england:
            mask = hospitals["Country"] == "England"
            hospitals = hospitals[mask]
        self.hospitals = hospitals

        # Load and parse LSOA travel matrix
        travel_matrix = pd.read_csv(
            "./data/lsoa_travel_time_matrix_calibrated.csv", index_col="LSOA"
        )

        ivt_hospitals = list(hospitals[hospitals["Use_IVT"] == 1]["Postcode"])
        self.lsoa_ivt_travel_time = dict(
            travel_matrix[ivt_hospitals].min(axis=1))
        self.lsoa_ivt_unit = dict(travel_matrix[ivt_hospitals].idxmin(axis=1))

        mt_hospitals = list(hospitals[hospitals["Use_MT"] == 1]["Postcode"])
        self.lsoa_mt_travel_time = dict(
            travel_matrix[mt_hospitals].min(axis=1))
        self.lsoa_mt_unit = dict(travel_matrix[mt_hospitals].idxmin(axis=1))

        # Load and parse inter_hospital travel time for MT
        inter_hospital_time = pd.read_csv(
            "./data/inter_hospital_time_calibrated.csv", index_col="from_postcode"
        )

        inter_hospital_time = inter_hospital_time.loc[ivt_hospitals][mt_hospitals]

        # Set distance between same hospitals to zero
        rows = list(inter_hospital_time.index)
        cols = list(inter_hospital_time)
        for row in rows:
            for col in cols:
                if row == col:
                    inter_hospital_time.loc[row][col] = 0

        self.mt_transfer_time = dict(inter_hospital_time.min(axis=1))
        self.mt_transfer_unit = dict(inter_hospital_time.idxmin(axis=1))
