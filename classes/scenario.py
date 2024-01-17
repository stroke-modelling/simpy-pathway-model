"""
Scenario class with global parameters for the pathway model.
"""
import numpy as np
import pandas as pd


class Scenario(object):
    """
    Global variables for model.

    class Scenario():

    Attributes
    ----------

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


    Methods
    -------
    load_data:
        Loads data to be used

    _load_hospitals:
        Loads data on the requested stroke teams.

    _load_admissions:
        Loads admissions data for the requested stroke teams and
        the LSOAs in their catchment areas.

    _load_lsoa_travel:
        Loads data on travel times from each LSOA to its nearest
        stroke units offering IVT and MT.

    _load_stroke_unit_travel:
        Loads data on travel times between stroke units.
    """

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""

        # Which LSOAs will we use?
        self.mt_hub_postcodes = []
        self.limit_to_england = True

        self.run_duration = 365  # Days
        self.warm_up = 50

        # Which stroke team choice model will we use?
        self.destination_decision_type = 0
        # 0 is 'drip-and-ship'

        # Are we using any extra units?
        # i.e. not used in the main IVT and MT units list.
        self.custom_units = False

        # What are the chances of treatment?
        self.probability_ivt = 1.0
        self.probability_mt = 1.0

        # Set process times.
        # Each tuple contains (minimum time, maximum time).
        # When both values are the same, all generated times
        # are that same value with no variation.
        self.process_time_call_ambulance = (30, 30)
        self.process_time_ambulance_response = (30, 30)
        self.process_ambulance_on_scene_duration = (20, 20)
        self.process_time_arrival_to_needle = (30, 30)
        self.process_time_arrival_to_puncture = (45, 45)
        self.transfer_time_delay = 30
        self.process_time_transfer_arrival_to_puncture = (60, 60)

        # Stroke unit services updates.
        # Change which units provide IVT, MT, and MSU by changing
        # their 'Use_IVT' flags in the services dataframe.
        # Example:
        # self.services_updates = {
        #     'hospital_name1': {'Use_MT': 0},
        #     'hospital_name2': {'Use_IVT': 0, 'Use_MSU': None},
        #     }
        self.services_updates = {}

        # Set up paths to files.
        # TO DO - check if output folder already exists,
        # make a new output folder for each run.
        self.paths_dict = dict(
            data_read_path='./data/',
            output_folder='./output/',
        )

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Convert run duration to minutes
        self.run_duration *= 1440

        # Load data:
        # (run this after MT hospitals are updated in
        # initial_data or kwargs).
        self.load_data()

    def load_data(self):
        """
        Load required data.

        Stores the following in the Globvars object:
        + hospitals
        + lsoa_ivt_travel_time
        + lsoa_ivt_unit
        + lsoa_mt_travel_time
        + lsoa_mt_unit
        + lsoa_names
        + lsoa_relative_frequency
        + lsoa_travel_time
        + mt_transfer_time
        + mt_transfer_unit
        + total_admissions

        More details on each attribute are given in the docstrings
        of the methods that create them.
        """
        # Import hospital names:
        self._load_hospitals()
        # Stores:
        # + self.hospitals

        # Find which LSOAs are in these stroke teams' catchment areas:
        self._load_lsoa_names()
        # Stores:
        # + self.lsoa_names

        # Stroke unit data for each LSOA.
        self._load_lsoa_travel()
        # Stores:
        # + self.lsoa_ivt_travel_time
        # + self.lsoa_ivt_unit
        # + self.lsoa_mt_travel_time
        # + self.lsoa_mt_unit

        # Import admissions statistics for those hospitals:
        self._load_admissions()
        # Stores:
        # + self.total_admissions
        # + self.lsoa_relative_frequency
        # + self.inter_arrival_time

        # Transfer stroke unit data.
        self._load_transfer_travel()
        # Stores:
        # + self.mt_transfer_time
        # + self.mt_transfer_unit

    # ################################
    # ##### NATIONAL INFORMATION #####
    # ################################
    def _set_national_hospital_services(self):
        """
        WIP

        Have some "national" units that always quietly exist
        even if we're not directly simulating them here,
        because otherwise patients from Newcastle will be sent
        to the nearest unit even if that's Cornwall.

        """
        # Load default stroke unit services:
        dir_input = self.paths_dict['data_read_path']
        services = pd.read_csv(
            f'{dir_input}stroke_unit_services.csv', index_col='Postcode')

        # Overwrite hospital info if given.
        # Keep the same list of hospitals nationally but update
        # which services they provide. We can't easily add a totally
        # new unit because the travel times need to be calculated
        # outside of this class.

        # Define "kv" to shorten following line:
        kv = zip(self.services_updates.keys(),
                 self.services_updates.values())
        for hospital, service_dict in kv:
            for key, value in service_dict:
                success = True
                try:
                    value = int(value)
                except TypeError:
                    if value is None:
                        # Nothing to see here.
                        pass
                    else:
                        # This shouldn't happen.
                        # TO DO - flag up an error or something?
                        success = False
                if success:
                    # Get the right row with services.loc[hospital],
                    # then the right column with [key],
                    # and overwrite the existing value.
                    services.loc[hospital][key] = value

        # Store national hospitals and their services in self.
        self.national_hospital_services = services

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'national_stroke_unit_services.csv'
        services.to_csv(f'{dir_output}{file_name}')

    def _find_national_lsoa_nearest_units(self):
        """
        WIP
        """
        # Load travel time matrix:
        df_time_lsoa_hospital = pd.read_csv(
            '../data_tabular/lsoa_travel_time_matrix_calibrated.csv',
            index_col='LSOA'
            )
        # Each column is a postcode of a stroke team and
        # each row is an LSOA name (LSOA11NM).

        # Get list of services that each stroke team provides:
        df_stroke_teams = self.national_hospital_services
        # Each row is a different stroke team and the columns are
        # 'Postcode', 'SSNAP name', 'Use_IVT', 'Use_MT', 'Use_MSU'
        # where the "Use_" columns contain 0 (False) or 1 (True).

        # Make masks of units offering each service:
        mask_ivt = df_stroke_teams['Use_IVT'] == 1
        mask_mt = df_stroke_teams['Use_MT'] == 1
        mask_msu = df_stroke_teams['Use_MSU'] == 1
        # Make lists of units offering each service:
        teams_ivt = df_stroke_teams['Postcode'][mask_ivt].values
        teams_mt = df_stroke_teams['Postcode'][mask_mt].values
        teams_msu = df_stroke_teams['Postcode'][mask_msu].values
        # Store these in a dict:
        teams_dict = dict(
            IVT=teams_ivt,
            MT=teams_mt,
            MSU=teams_msu,
        )

        # Define functions for finding the nearest stroke team
        # to each LSOA and copying over the useful information.
        # These functions will be called once for each of the
        # list of teams in the teams_dict.
        def _find_nearest_units(
                df_time_lsoa_hospital: pd.DataFrame,
                teams: list,
                label: str,
                df_results: pd.DataFrame = pd.DataFrame()
                ):
            """
            WIP

            Index must be LSOA

            Add these columns to the DataFrame:
            time_nearest_{label}
            postcode_nearest_{label}
            """
            # The smallest time in each row:
            df_results[f'time_nearest_{label}'] = (
                df_time_lsoa_hospital[teams].min(axis='columns'))
            # The name of the column containing the smallest
            # time in each row:
            df_results[f'postcode_nearest_{label}'] = (
                df_time_lsoa_hospital[teams].idxmin(axis='columns'))

            return df_results

        def _merge_unit_info(
                df_results: pd.DataFrame,
                df_stroke_teams: pd.DataFrame,
                label: str
                ):
            """
            WIP

            Index must be LSOA name

            Add these columns to the DataFrame:
            ssnap_name_nearest_{label}
            """
            # Why is this here? TO DO
            df_results['lsoa'] = df_results.index

            # Merge in other info about the nearest units:
            df_results = pd.merge(
                df_results,
                df_stroke_teams[['Postcode', 'SSNAP name']],
                left_on=f'postcode_nearest_{label}',
                right_on='Postcode'
            )
            # Remove the repeat column:
            df_results = df_results.drop('Postcode', axis=1)
            # Rename columns:
            df_results = df_results.rename(columns={
                'SSNAP name': f'ssnap_name_nearest_{label}',
            })
            return df_results

        # Run these functions for the groups of stroke units.
        # Put the results in this dataframe where each row
        # is a different LSOA:
        df_results = pd.DataFrame(index=df_time_lsoa_hospital.index)
        # Fill in the nearest stroke unit info:
        for label, teams in zip(teams_dict.keys(), teams_dict.values()):
            df_results = _find_nearest_units(
                df_time_lsoa_hospital, teams, label, df_results)
            df_results = _merge_unit_info(
                df_results, df_stroke_teams, label)

        # Load data on LSOA names, codes, regions...
        df_regions = pd.read_csv('../data_tabular/LSOA_regions.csv')
        # Each row is a different LSOA and the columns include
        # LSOA11NM, LSOA11CD, longitude and latitude, and larger
        # regional groupings (e.g. Clinical Care Group names).

        # Add in extra identifiers - LSOA11CD from ONS data.
        df_results = pd.merge(
            df_results,
            df_regions[['LSOA11NM', 'LSOA11CD']],
            left_on='lsoa',
            right_on='LSOA11NM'
        )
        # Remove the repeat column:
        df_results = df_results.drop('lsoa', axis=1)

        # Reorder columns:
        cols_order = ['LSOA11NM', 'LSOA11CD']
        for label in list(teams_dict.keys()):
            cols_order += [
                f'time_nearest_{label}',
                f'postcode_nearest_{label}',
                f'ssnap_name_nearest_{label}'
                ]
        df_results = df_results[cols_order]

        # Save this to self.
        self.national_lsoa_nearest_units = df_results

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'national_travel_lsoa_stroke_units.csv'
        df_results.to_csv(f'{dir_output}{file_name}')

    def _find_national_mt_catchment_areas(self):
        """
        Find catchment areas for national hospitals offering MT.

        All hospitals in data, not just those selected for simulation.

        Wheel-and-spoke model.
        """
        # Get list of services that each stroke team provides:
        df_stroke_teams = self.national_hospital_services
        # Each row is a different stroke team and the columns are
        # 'Postcode', 'SSNAP name', 'Use_IVT', 'Use_MT', 'Use_MSU'
        # where the "Use_" columns contain 0 (False) or 1 (True).

        # Travel time matrix between hospitals:
        df_time_inter_hospital = pd.read_csv(
            '../data_tabular/inter_hospital_time_calibrated.csv',
            index_col='from_postcode'
            )

        # Pick out the names of hospitals offering MT:
        mask_mt = (df_stroke_teams['Use_MT'] == 1)
        mt_hospital_names = df_stroke_teams['Postcode'][mask_mt].values
        # Reduce columns of inter-hospital time matrix to just MT hospitals:
        df_time_inter_hospital = df_time_inter_hospital[mt_hospital_names]

        # From this reduced dataframe, pick out
        # the smallest time in each row and
        # the MT hospital that it belongs to.
        # Store the results in this DataFrame:
        df_nearest_mt = pd.DataFrame(index=df_time_inter_hospital.index)
        # The smallest time in each row:
        df_nearest_mt['time_nearest_MT'] = (
            df_time_inter_hospital.min(axis='columns'))
        # The name of the column containing the smallest time in each row:
        df_nearest_mt['name_nearest_MT'] = (
            df_time_inter_hospital.idxmin(axis='columns'))

        # Store in self:
        self.national_inter_hospital_time = df_nearest_mt

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'national_stroke_unit_nearest_mt.csv'
        df_nearest_mt.to_csv(f'{dir_output}{file_name}')

    # ##########################
    # ##### SELECTED UNITS #####
    # ##########################
    def _load_hospitals(self):
        """
        Load data on the selected stroke units.

        If no units are specified but "limit_to_england" is True,
        then only English stroke units are kept. If no units are
        specified and "limit_to_england" is False, then all stroke
        units are kept.

        Stores
        ------

        hospitals:
            pd.DataFrame. Each stroke team's data including name,
            postcode, region, lat/long, provide IVT or MT...
        """
        # Load and parse hospital data
        hospitals = pd.read_csv("./data/stroke_hospitals_2022.csv")
        # Only keep stroke units that offer IVT and/or MT:
        hospitals["Use"] = hospitals[["Use_IVT", "Use_MT"]].max(axis=1)
        mask = hospitals["Use"] == 1
        hospitals = hospitals[mask]

        # Limit the available hospitals if required.
        if len(self.mt_hub_postcodes) > 0:
            # If a list of MT units was given, use only those units.

            # Which IVT units are feeder units for these MT units?
            # These lines also pick out the MT units themselves.
            df_feeders = pd.read_csv('./data/nearest_mt_each_hospital.csv')
            # For each MT unit, find whether each stroke unit has this
            # as its nearest MT unit. Get a long list of True/False
            # values for each MT unit.
            feeders_bool = [
                df_feeders['name_nearest_MT'].str.contains(s)
                for s in self.mt_hub_postcodes
                ]
            # Make a mask that is True for any stroke unit that
            # answered True to any of the feeders_bool lists,
            # i.e. its nearest MT unit is in the requested list.
            mask = np.any(feeders_bool, axis=0)
            # Limit the feeders dataframe to just those stroke teams
            # to get a list of postcodes of feeder units.
            df_feeders = df_feeders[mask]
            # Limit the "hospitals" dataframe to these feeder postcodes:
            hospitals = pd.merge(
                left=hospitals,
                right=df_feeders,
                left_on='Postcode',
                right_on='from_postcode',
                how='inner'
            )[hospitals.columns]
        elif self.limit_to_england:
            # Limit the data to English stroke units only.
            mask = hospitals["Country"] == "England"
            hospitals = hospitals[mask]
        else:
            # Use the full "hospitals" data.
            pass
        self.hospitals = hospitals

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'selected_stroke_units.csv'
        hospitals.to_csv(f'{dir_output}{file_name}')

    def _load_lsoa_names(self):
        """
        WIP
        Load names of LSOAs in catchment areas of chosen stroke teams.

        Stores
        ------
        lsoa_names:
            np.array. Names of all LSOAs considered.
        """
        # Take list of all LSOA names and travel times:
        df_travel = self.national_lsoa_nearest_units
        # This has one row for each LSOA nationally and columns
        # for LSOA name and ONS code (LSOA11NM and LSOA11CD),
        # and time, postcode, and SSNAP name of the nearest unit
        # for each unit type (IVT, MT, MSU).
        # Columns:
        # + LSOA11NM
        # + LSOA11CD
        # + time_nearest_IVT
        # + postcode_nearest_IVT
        # + ssnap_name_nearest_IVT
        # + time_nearest_MT
        # + postcode_nearest_MT
        # + ssnap_name_nearest_MT
        # + time_nearest_MSU
        # + postcode_nearest_MSU
        # + ssnap_name_nearest_MSU

        # Limit the available hospitals if required.
        if len(self.mt_hub_postcodes) > 0:
            # If a list of MT units was given, limit the admissions
            # data to just those MT units and feeder IVT units
            # that we saved earlier as self.hospitals.

            # Which LSOAs are in the catchment areas for these IVT units?
            # For each stroke team, make a long list of True/False for
            # whether each LSOA has this as its nearest unit.
            postcode_cols = [
                'postcode_nearest_IVT',
                'postcode_nearest_MT',
                'postcode_nearest_MSU',
            ]
            lsoa_bool = [
                df_travel[col].str.contains(s)
                for col in postcode_cols
                for s in self.hospitals['Postcode'].values
                ]
            # Mask is True for any LSOA that is True in any of the
            # lists in lsoa_bool.
            mask = np.any(lsoa_bool, axis=0)
            # Limit the data to just these LSOAs:
            lsoas_to_include = df_travel['LSOA11NM'][mask]
        elif self.limit_to_england:
            # Limit the data to English LSOAs only.
            # The LSOA11CD (ONS code for each LSOA) begins with
            # an "E" for English and "W" for Welsh LSOAs.
            # All other characters are numbers.
            mask_england = df_travel['LSOA11CD'].str.contains('E')
            lsoas_to_include = df_travel['LSOA11NM'][mask_england]
        else:
            # Just use all LSOAs in the file.
            lsoas_to_include = df_travel['LSOA11NM']

        # Store in self:
        self.lsoa_names = lsoas_to_include

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'selected_lsoas.csv'
        lsoas_to_include.to_csv(f'{dir_output}{file_name}')

    def _load_admissions(self):
        """
        Load admission data on the selected stroke teams.

        If no units are specified but "limit_to_england" is True,
        then only English stroke units are kept. If no units are
        specified and "limit_to_england" is False, then all stroke
        units are kept.

        Stores
        ------

        total_admissions:
            float. Total admissions in a year across selected
            stroke units.

        lsoa_relative_frequency:
            np.array. Relative frequency of each considered LSOA
            in the admissions data. Same order as self.lsoa_names.

        lsoa_names:
            np.array. Names of all LSOAs considered.
            Same order as lsoa_relative_frequency.

        inter_arrival_time:
            float. Average time between admissions in the
            considered stroke teams.
        """
        # Load and parse admissions data
        admissions = pd.read_csv("./data/admissions_2017-2019.csv")

        # Keep only these LSOAs in the admissions data:
        admissions = pd.merge(
            left=admissions,
            right=self.lsoa_names,
            left_on='area',
            right_on='LSOA11NM',
            how='inner'
        )

        # Process admissions.
        # Total admissions across these hospitals in a year:
        self.total_admissions = np.round(admissions["Admissions"].sum(), 0)
        # Relative frequency of admissions across a year:
        self.lsoa_relative_frequency = np.array(
            admissions["Admissions"] / self.total_admissions
        )
        self.lsoa_names = list(admissions["area"])
        # Average time between admissions to these hospitals in a year:
        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions

    def _load_lsoa_travel(self):
        """
        WIP
        Stroke unit data for each LSOA.

        Stores
        ------

        lsoa_ivt_travel_time:
            dict. Each LSOA's nearest IVT unit travel time.

        lsoa_ivt_unit:
            dict. Each LSOA's nearest IVT unit name.

        lsoa_mt_travel_time:
            dict. Each LSOA's nearest MT unit travel time.

        lsoa_mt_unit:
            dict. Each LSOA's nearest MT unit name.
        """
        # Use the list of LSOA names to include:
        lsoa_names = self.lsoa_names

        # Take list of all LSOA names and travel times:
        df_travel = self.national_lsoa_nearest_units
        # This has one row for each LSOA nationally and columns
        # for LSOA name and ONS code (LSOA11NM and LSOA11CD),
        # and time, postcode, and SSNAP name of the nearest unit
        # for each unit type (IVT, MT, MSU).
        # Columns:
        # + LSOA11NM
        # + LSOA11CD
        # + time_nearest_IVT
        # + postcode_nearest_IVT
        # + ssnap_name_nearest_IVT
        # + time_nearest_MT
        # + postcode_nearest_MT
        # + ssnap_name_nearest_MT
        # + time_nearest_MSU
        # + postcode_nearest_MSU
        # + ssnap_name_nearest_MSU

        # Limit the big DataFrame to just the LSOAs wanted:
        df_travel = pd.merge(
            df_travel,
            pd.DataFrame(lsoa_names, columns=['LSOA11NM']),
            left_on='LSOA11NM',
            right_on='LSOA11NM'
            )

        # Separate out the columns and store in self:
        self.lsoa_ivt_travel_time = dict(df_travel['time_nearest_IVT'])
        self.lsoa_ivt_unit = dict(df_travel['postcode_nearest_IVT'])
        self.lsoa_mt_travel_time = dict(df_travel['time_nearest_MT'])
        self.lsoa_mt_unit = dict(df_travel['postcode_nearest_MT'])
        self.lsoa_msu_travel_time = dict(df_travel['time_nearest_MSU'])
        self.lsoa_msu_unit = dict(df_travel['postcode_nearest_MSU'])

    def _load_transfer_travel(self):
        """
        Data for the transfer stroke unit of each selected stroke unit.

        Stores
        ------

        mt_transfer_time:
            dict. Each stroke unit's travel time to their nearest
            MT transfer unit.

        mt_transfer_unit:
            dict. Each stroke unit's nearest MT transfer unit's name.
        """
        # Load and parse inter hospital travel time for MT
        # inter_hospital_time = pd.read_csv(
        #     "./data/inter_hospital_time_calibrated.csv",
        #     index_col="from_postcode"
        # )
        inter_hospital_time = self.national_inter_hospital_time

        ivt_hospitals = list(
            self.hospitals[self.hospitals["Use_IVT"] == 1]["Postcode"])
        mt_hospitals = list(
            self.hospitals[self.hospitals["Use_MT"] == 1]["Postcode"])
        inter_hospital_time = (
            inter_hospital_time.loc[ivt_hospitals][mt_hospitals])
        # Take the MT transfer unit as being the closest one
        # (i.e. minimum travel time) to each IVT unit.
        self.mt_transfer_time = dict(inter_hospital_time.min(axis=1))
        self.mt_transfer_unit = dict(inter_hospital_time.idxmin(axis=1))
