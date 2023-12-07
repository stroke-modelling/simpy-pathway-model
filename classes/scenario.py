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

        # Which LSOAs will we use?
        self.mt_hub_postcodes = []
        self.limit_to_england = True

        self.run_duration = 365 #Days
        self.warm_up = 50


        # Set process times
        self.process_time_ambulance_response = (10, 40)
        self.process_time_call_ambulance = (5, 60)
        # Lognorm mu and sigma parameters:
        self.process_time_arrival_to_scan = (np.NaN, np.NaN)
        self.process_time_scan_to_needle = (np.NaN, np.NaN)

        # Overwrite default values (can take named arguments or a dictionary)

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Convert run duration to minutes
        self.run_duration *= 1440

        # Load data
        # (after MT hospitals are updated)
        self.load_data()

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
        # Load and parse hospital data
        hospitals = pd.read_csv("./data/stroke_hospitals_2022.csv")
        hospitals["Use"] = hospitals[["Use_IVT", "Use_MT"]].max(axis=1)
        mask = hospitals["Use"] == 1
        hospitals = hospitals[mask]
        if len(self.mt_hub_postcodes) > 0:
            # If given, use only these MT units.

            # Which IVT units are feeder units for these MT units?
            df_feeders = pd.read_csv('./data/nearest_mt_each_hospital.csv')
            df_feeders = df_feeders[
                np.any([df_feeders['name_nearest_MT'].str.contains(s) for s in self.mt_hub_postcodes], axis=0)
                ]
            # Match these postcodes to the "hospitals" dataframe:
            hospitals = pd.merge(
                left=hospitals,
                right=df_feeders,
                left_on='Postcode',
                right_on='from_postcode',
                how='inner'
            )[hospitals.columns]
        elif self.limit_to_england:
            mask = hospitals["Country"] == "England"
            hospitals = hospitals[mask]
        else:
            # Use the full "hospitals" data.
            pass
        self.hospitals = hospitals

        # Load and parse admissions data
        admissions = pd.read_csv("./data/admissions_2017-2019.csv")

        if len(self.mt_hub_postcodes) > 0:
            # Which LSOAs are in the catchment areas for these IVT units?
            df_nearest_teams = pd.read_csv('./data/lsoa_nearest_stroke_team.csv')
            df_nearest_teams = df_nearest_teams[
                np.any(
                    [df_nearest_teams[
                        'postcode_nearest_stroke_team'].str.contains(s)
                     for s in hospitals['Postcode'].values],
                     axis=0
                    )
                ]
            lsoas_to_include = df_nearest_teams['LSOA11NM']

            # Keep only these LSOAs in the admissions data:
            admissions = pd.merge(
                left=admissions,
                right=lsoas_to_include,
                left_on='area',
                right_on='LSOA11NM',
                how='inner'
            )
        elif self.limit_to_england:
            # If no MT unit names were specified:
            mask = admissions["England"] == 1
            admissions = admissions[mask]
        else:
            # Just use all LSOAs in the file.
            pass

        self.total_admissions = np.round(admissions["Admissions"].sum(), 0)
        self.lsoa_relative_frequency = np.array(
            admissions["Admissions"] / self.total_admissions
        )
        self.lsoa_names = list(admissions["area"])

        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions


        # Load and parse LSOA travel matrix
        df_travel = pd.read_csv(
            "./data/lsoa_nearest_stroke_team.csv", index_col="LSOA11NM"
        )
        self.lsoa_ivt_travel_time = dict(df_travel['time_nearest_stroke_team'])
        self.lsoa_ivt_unit = dict(df_travel['postcode_nearest_stroke_team'])

        self.lsoa_mt_travel_time = dict(df_travel['time_nearest_MT'])
        self.lsoa_mt_unit = dict(df_travel['postcode_nearest_MT'])

        # Load and parse inter_hospital travel time for MT
        inter_hospital_time = pd.read_csv(
            "./data/inter_hospital_time_calibrated.csv", index_col="from_postcode"
        )
        ivt_hospitals = list(hospitals[hospitals["Use_IVT"] == 1]["Postcode"])
        mt_hospitals = list(hospitals[hospitals["Use_MT"] == 1]["Postcode"])
        inter_hospital_time = inter_hospital_time.loc[ivt_hospitals][mt_hospitals]

        self.mt_transfer_time = dict(inter_hospital_time.min(axis=1))
        self.mt_transfer_unit = dict(inter_hospital_time.idxmin(axis=1))


        # Load in hospital performance data
        self.hospital_performance = pd.read_csv(
            './data/hospital_performance.csv', index_col='Postcode'
        )
