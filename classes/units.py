"""
Units class for defining choice of stroke unit through the pathway.
"""
import numpy as np
import pandas as pd


class Units(object):
    """
    Links between stroke units.
    """

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""
        # Which stroke team choice model will we use?
        self.destination_decision_type = 0
        # 0 is 'drip-and-ship'

        # Are we using any extra units?
        # i.e. not used in the main IVT and MT units list.
        self.custom_units = False

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

        # Load data:
        # (run this after MT hospitals are updated in
        # initial_data or kwargs).
        self.load_data()

    def load_data(self):
        """
        Load required data.

        Stores the following in the Globvars object:
        + national_hospital_services
        + national_lsoa_nearest_units
        + national_ivt_feeder_units
        + national_mt_transfer_time
        + national_mt_transfer_unit

        More details on each attribute are given in the docstrings
        of the methods that create them.
        """
        # ##### NATIONAL INFORMATION #####
        # Load information about all stroke units nationally,
        # whether they're being highlighted in the simulation or not.
        # Find which stroke units provide IVT, MT, and MSU:
        self._set_national_hospital_services()
        # Stores:
        # + self.national_hospital_services

        # Find each LSOA's nearest IVT, MT, and MSU units:
        self._find_national_lsoa_nearest_units()
        # Stores:
        # + self.national_lsoa_nearest_units

        # Find which IVT units are feeders to each MT unit:
        self._find_national_mt_feeder_units()
        # Stores:
        # + self.national_ivt_feeder_units

        # Transfer stroke unit data.
        self._find_national_transfer_travel()
        # Stores:
        # + self.national_mt_transfer_time
        # + self.national_mt_transfer_unit

        # Place everything useful into a dict for returning:
        national_dict = dict(
            hospital_services=self.national_hospital_services,
            lsoa_nearest_units=self.national_lsoa_nearest_units,
            ivt_feeder_units=self.national_ivt_feeder_units,
            mt_transfer_time=self.national_mt_transfer_time,
            mt_transfer_unit=self.national_mt_transfer_unit,
        )

        return national_dict

    # ################################
    # ##### NATIONAL INFORMATION #####
    # ################################
    def _set_national_hospital_services(self):
        """
        Make table of which stroke units provide which treatments.

        Each stroke unit has a flag in this table for each of:
        + Use_IVT
        + Use_MT
        + Use_MSU
        The value is set to either 0 (not provided) or 1 (provided).

        Most of the values are stored in a reference file but
        they can be updated by the user with the dictionary
        self.services_updates.

        These values should be set for all units nationally,
        because otherwise patients from e.g. Newcastle will have their
        nearest stroke unit set to e.g. Cornwall.

        Stores
        ------

        national_hospital_services:
            pd.DataFrame. Each stroke team's services provided.
            Columns for whether a team provides IVT, MT, and MSU.
        """
        # Load default stroke unit services:
        dir_input = self.paths_dict['data_read_path']
        services = pd.read_csv(
            f'{dir_input}stroke_unit_services.csv',
            index_col='Postcode'
            )
        # Each row is a stroke unit. The columns are 'Postcode' and
        # 'SSNAP name' (str), and 'Use_IVT', 'Use_MT', and 'Use_MSU'
        # (int | bool).

        # Overwrite hospital info if given.
        # Keep the same list of hospitals nationally but update
        # which services they provide. We can't easily add a totally
        # new unit because the travel times need to be calculated
        # outside of this class.

        # Define "kv" to shorten following line:
        kv = zip(self.services_updates.keys(),
                 self.services_updates.values())
        for hospital, service_dict in kv:
            for key, value in zip(service_dict.keys(), service_dict.values()):
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
                    services.loc[hospital, key] = value

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'national_stroke_unit_services.csv'
        services.to_csv(f'{dir_output}{file_name}')

        # Remove index column:
        services = services.reset_index()

        # Store national hospitals and their services in self.
        self.national_hospital_services = services

    def _find_national_lsoa_nearest_units(self):
        """
        Find each LSOA's nearest stroke units providing each service.

        Find the name, postcode and travel time to the nearest
        stroke unit providing each of IVT, MT, and an MSU.

        Stores
        ------

        national_lsoa_nearest_units:
            pd.DataFrame. One row for each LSOA nationally
            and columns containing the nearest units providing
            IVT, MT, and MSU for each LSOA and the travel times.
        """
        # Load travel time matrix:
        df_time_lsoa_hospital = pd.read_csv(
            './data/lsoa_travel_time_matrix_calibrated.csv',
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
            Find the nearest units from the travel time matrix.

            Index must be LSOA names.

            Inputs
            ------
            df_time_lsoa_hospital:
                pd.DataFrame. Travel time matrix between LSOAs
                and stroke units.
            teams:
                list. List of teams for slicing the travel time
                DataFrame, only consider a subset of teams.
            label:
                str. A label for the resulting columns.
            df_results:
                pd.DataFrame. The DataFrame to store results in.
                If none is given, a new one is created.

            Result
            ------
            Add these columns to the DataFrame:
            time_nearest_{label}
            postcode_nearest_{label}
            """
            if (df_results.index != df_time_lsoa_hospital.index).any():
                # If a new dataframe was made, make sure the
                # index column contains the LSOA names.
                df_results.index = df_time_lsoa_hospital.index
            else:
                pass
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

            Inputs
            ------
            df_results:
                pd.DataFrame. Contains columns for nearest stroke unit
                and travel time from each LSOA. New results here will
                be stored in this DataFrame.
            df_stroke_teams:
                pd.DataFrame. Contains information on the stroke units
                such as their region and SSNAP name.
            label:
                str. A label for the resulting columns.

            Result
            ------
            Add these columns to the DataFrame:
            ssnap_name_nearest_{label}
            """
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

            df_results = df_results.set_index('lsoa')
            return df_results

        # Run these functions for the groups of stroke units.
        # Put the results in this dataframe where each row
        # is a different LSOA:
        df_results = pd.DataFrame(index=df_time_lsoa_hospital.index)
        # Fill in the nearest stroke unit info:
        for label, teams in zip(teams_dict.keys(), teams_dict.values()):
            df_results = _find_nearest_units(
                df_time_lsoa_hospital.copy(), teams, label, df_results)
            df_results = _merge_unit_info(
                df_results, df_stroke_teams, label)

        # Load data on LSOA names, codes, regions...
        df_regions = pd.read_csv('./data/lsoa_to_msoa.csv')
        # Each row is a different LSOA and the columns include
        # LSOA11NM, LSOA11CD, longitude and latitude, and larger
        # regional groupings (e.g. Clinical Care Group names).

        # Add in extra identifiers - LSOA11CD from ONS data.
        df_results = pd.merge(
            df_results,
            df_regions[['lsoa11nm', 'lsoa11cd']],
            left_on='lsoa',
            right_on='lsoa11nm'
        )
        # # Remove the repeat column:
        # df_results = df_results.drop('lsoa', axis=1)
        # Rename columns:
        df_results = df_results.rename(columns={
            'lsoa11nm': 'LSOA11NM',
            'lsoa11cd': 'LSOA11CD',
            })
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

    def _find_national_mt_feeder_units(self):
        """
        Find catchment areas for national hospitals offering MT.

        For each stroke unit, find the name of and travel time to
        its nearest MT unit. Wheel-and-spoke model. If the unit
        is an MT unit then the travel time is zero.

        Stores
        ------

        national_ivt_feeder_units:
            pd.DataFrame. Each row is a stroke unit. Columns are
            its postcode, the postcode of the nearest MT unit,
            and travel time to that MT unit.
        """
        # Get list of services that each stroke team provides:
        df_stroke_teams = self.national_hospital_services
        # Each row is a different stroke team and the columns are
        # 'Postcode', 'SSNAP name', 'Use_IVT', 'Use_MT', 'Use_MSU'
        # where the "Use_" columns contain 0 (False) or 1 (True).

        # Travel time matrix between hospitals:
        df_time_inter_hospital = pd.read_csv(
            './data/inter_hospital_time_calibrated.csv',
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
        self.national_ivt_feeder_units = df_nearest_mt

        # Save output to output folder.
        dir_output = self.paths_dict['output_folder']
        file_name = 'national_stroke_unit_nearest_mt.csv'
        df_nearest_mt.to_csv(f'{dir_output}{file_name}')

    def _find_national_transfer_travel(self):
        """
        Data for the transfer stroke unit of each national stroke unit.

        Stores
        ------

        national_mt_transfer_time:
            dict. Each stroke unit's travel time to their nearest
            MT transfer unit.

        national_mt_transfer_unit:
            dict. Each stroke unit's nearest MT transfer unit's name.
        """
        # Load and parse inter hospital travel time for MT
        inter_hospital_time = self.national_ivt_feeder_units

        self.national_mt_transfer_time = dict(
            inter_hospital_time['time_nearest_MT'])
        self.national_mt_transfer_unit = dict(
            inter_hospital_time['name_nearest_MT'])
