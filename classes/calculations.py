"""
Units class for defining choice of stroke unit through the pathway.

TO DO - should this be a functions file?
don't want a big run() function here - have clear names for what it actually does.
Yes - the only self. remaining are Setup, can replace with direct paths/filenames as kwargs.
Plus read in files using import_relative or whatever, 
do it in the Scenario class, pass the df to here as arg.

"""
import pandas as pd
import os  # For checking directory existence

from classes.setup import Setup


class Calculations(object):
    """
    Links between stroke units.
    """

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # If no setup was given, create one now:
        try:
            self.setup
        except AttributeError:
            self.setup = Setup()

        # Load data:
        # (run this after MT hospitals are updated in
        # initial_data or kwargs).
        # self.load_data()

    # def load_data(self):
    #     """
    #     Load required data.

    #     More details on each attribute are given in the docstrings
    #     of the methods that create them.
    #     """
    #     # ##### NATIONAL INFORMATION #####
    #     # Load information about all stroke units nationally,
    #     # whether they're being highlighted in the simulation or not.
    #     # Find which stroke units provide IVT, MT, and MSU:
    #     self._set_national_hospital_services()
    #     # Stores:
    #     # + self.national_hospital_services
    #     #   --> saves to: file_national_unit_services

    #     # Find each LSOA's nearest IVT, MT, and MSU units:
    #     # self._find_national_lsoa_nearest_units()  # TEMP COMMENTED OUT - FIX ME PLEASE - TO DO
    #     # Stores:
    #     # + self.national_lsoa_nearest_units
    #     #   --> saves to: file_national_lsoa_travel

    #     # Find which IVT units are feeders to each MT unit:
    #     self._find_national_mt_feeder_units()
    #     # Stores:
    #     # + self.national_ivt_feeder_units
    #     #   --> saves to: file_national_transfer_units

    #     # Transfer stroke unit data.
    #     self._find_national_transfer_travel()
    #     # Stores:
    #     # + self.national_mt_transfer_time
    #     # + self.national_mt_transfer_unit
    #     #   --> no file saved.

    #     # Place everything useful into a dict for returning:
    #     national_dict = dict(
    #         hospital_services=self.national_hospital_services,
    #         # lsoa_nearest_units=self.national_lsoa_nearest_units,
    #         ivt_feeder_units=self.national_ivt_feeder_units,
    #         mt_transfer_time=self.national_mt_transfer_time,
    #         mt_transfer_unit=self.national_mt_transfer_unit,
    #     )

    #     return national_dict

    # ################
    # ##### LSOA #####
    # ################
    def find_each_lsoa_chosen_unit(self, df_time_lsoa_to_units):
        """

        """
        # Put the results in this dataframe where each row
        # is a different LSOA:
        df_results = pd.DataFrame(index=df_time_lsoa_to_units.index)
        # The smallest time in each row:
        df_results['time_nearest'] = (
            df_time_lsoa_to_units.min(axis='columns'))
        # The name of the column containing the smallest
        # time in each row:
        df_results['unit_postcode'] = (
            df_time_lsoa_to_units.idxmin(axis='columns'))
        return df_results

    def find_lsoa_catchment(
            self,
            teams_to_limit=[]
            ):
        # Load travel time matrix:
        path_to_file = os.path.join(self.setup.dir_reference_data,
                                    self.setup.file_input_travel_times)
        df_time_lsoa_to_units = pd.read_csv(path_to_file, index_col='LSOA')
        # Each column is a postcode of a stroke team and
        # each row is an LSOA name (LSOA11NM).

        # Limit columns to requested units:
        if len(teams_to_limit) > 0:
            df_time_lsoa_to_units = df_time_lsoa_to_units[teams_to_limit]

        # Assign LSOA by catchment area of these stroke units.
        df_catchment = self.find_each_lsoa_chosen_unit(
            df_time_lsoa_to_units)
        return df_catchment

    def limit_lsoa_catchment_to_selected_units(
            self,
            df_catchment,
            regions_selected,
            regions_to_limit=[],
            limit_to_england=False,
            limit_to_wales=False
            ):
        # Load in all LSOA names, codes, regions...
        path_to_file = os.path.join(self.setup.dir_reference_data,
                                    self.setup.file_input_lsoa_regions)
        df_lsoa = pd.read_csv(path_to_file)
        # Columns: [lsoa, lsoa_code, region_code, region, region_type,
        #           icb_code, icb, isdn]
        # If requested, limit to England or Wales.
        if limit_to_england:
            df_lsoa = df_lsoa[df_lsoa['region_type'] == 'SICBL']
        elif limit_to_wales:
            df_lsoa = df_lsoa[df_lsoa['region_type'] == 'LHB']

        # Keep a copy of the original catchment columns for later:
        cols_df_catchment = df_catchment.columns
        # Merge in region information to catchment:
        df_catchment.reset_index(inplace=True)
        df_catchment = pd.merge(
            df_catchment, df_lsoa,
            left_on='LSOA', right_on='lsoa', how='left'
        )
        df_catchment.drop('lsoa', axis='columns', inplace=True)
        df_catchment.set_index('LSOA', inplace=True)

        # Limit rows to LSOA in requested regions:
        if len(regions_to_limit) > 0:
            mask = df_catchment['region_code'].isin(regions_to_limit)
            df_catchment = df_catchment.loc[mask].copy()

        # Find where the results data is in selected regions:
        mask = df_catchment['region_code'].isin(regions_selected)
        # Find list of units catching any LSOA in selected regions:
        units_catching_lsoa = sorted(list(set(
            df_catchment.loc[mask]['unit_postcode'])))

        # Limit the results to only
        # LSOAs that are caught by units
        # that catch any LSOA in the selected regions.
        mask = df_catchment[
            'unit_postcode'].isin(units_catching_lsoa)
        df_catchment = df_catchment.loc[mask].copy()

        # Restore the shortened catchment DataFrame to its starting columns
        # plus the useful regions:
        cols = cols_df_catchment + ['region', 'region_code', 'region_type']
        df_catchment = df_catchment[cols]

        return df_catchment

    def find_catchment_info_regions_and_units(
            self, df_catchment, df_units_regions
            ):
        """
        """
        # Find list of regions containing LSOA caught by selected units.
        regions_containing_lsoa = sorted(list(set(
            df_catchment['region_code'])))

        # Find list of units catching any LSOA in selected regions:
        units_catching_lsoa = sorted(list(set(
            df_catchment['unit_postcode'])))

        # Limit the units data:
        mask = df_units_regions['Postcode'].isin(units_catching_lsoa)
        df_units_regions = df_units_regions[mask]
        # Find list of regions containing these units:
        regions_containing_units_catching_lsoa = (
            df_units_regions['region_code'].tolist())

        to_return = (
            regions_containing_lsoa,
            units_catching_lsoa,
            regions_containing_units_catching_lsoa
        )
        return to_return

    # #################
    # ##### UNITS #####
    # #################
    # def _set_national_hospital_services(self):
    #     """
    #     DELETE THIS

    #     Make table of which stroke units provide which treatments.

    #     Each stroke unit has a flag in this table for each of:
    #     + use_ivt
    #     + use_mt
    #     + use_msu
    #     The value is set to either 0 (not provided) or 1 (provided).

    #     Most of the values are stored in a reference file but
    #     they can be updated by the user with the ... UPDATE ME ------------------

    #     These values should be set for all units nationally,
    #     because otherwise patients from e.g. Newcastle will have their
    #     nearest stroke unit set to e.g. Cornwall.

    #     Stores
    #     ------

    #     national_hospital_services:
    #         pd.DataFrame. Each stroke team's services provided.
    #         Columns for whether a team provides IVT, MT, and MSU.
    #     """
    #     # Load default stroke unit services:
    #     dir_input = self.setup.dir_output_pathway
    #     file_input = self.setup.file_selected_units
    #     path_to_file = os.path.join(dir_input, file_input)
    #     services = pd.read_csv(path_to_file)
    #     # Each row is a stroke unit. The columns are 'postcode' and
    #     # 'SSNAP name' (str), and 'use_ivt', 'use_mt', and 'use_msu'
    #     # (int | bool).

    #     # Store national hospitals and their services in self.
    #     self.national_hospital_services = services

    #     # # Save output to output folder.
    #     # dir_output = self.setup.dir_output
    #     # file_name = self.setup.file_national_unit_services
    #     # path_to_file = os.path.join(dir_output, file_name)
    #     # services.to_csv(path_to_file, index=False)

    def find_national_mt_feeder_units(self, df_stroke_teams):
        """
        Find catchment areas for national hospitals offering MT.

        For each stroke unit, find the name of and travel time to
        its nearest MT unit. Wheel-and-spoke model. If the unit
        is an MT unit then the travel time is zero.

        TO DO - sort out whether this saves or not or repeats with the other one or what.

        Stores
        ------

        national_ivt_feeder_units:
            pd.DataFrame. Each row is a stroke unit. Columns are
            its postcode, the postcode of the nearest MT unit,
            and travel time to that MT unit.
        """
        # Get list of services that each stroke team provides:
        # df_stroke_teams = self.national_hospital_services
        # Each row is a different stroke team and the columns are
        # 'postcode', 'SSNAP name', 'use_ivt', 'use_mt', 'use_msu'
        # where the "use_" columns contain 0 (False) or 1 (True).

        # Pick out the names of hospitals offering IVT:
        mask_ivt = (df_stroke_teams['use_ivt'] == 1)
        ivt_hospital_names = df_stroke_teams['postcode'][mask_ivt].values
        # Pick out the names of hospitals offering MT:
        mask_mt = (df_stroke_teams['use_mt'] == 1)
        mt_hospital_names = df_stroke_teams['postcode'][mask_mt].values

        # Firstly, determine MT feeder units based on travel time.
        # Each stroke unit will be assigned the MT unit that it is
        # closest to in travel time.
        # Travel time matrix between hospitals:
        path_to_file = os.path.join(
            self.setup.dir_reference_data,
            self.setup.file_input_travel_times_inter_unit
            )
        df_time_inter_hospital = pd.read_csv(path_to_file,
                                             index_col='from_postcode')
        # Reduce columns of inter-hospital time matrix to just MT hospitals:
        df_time_inter_hospital = df_time_inter_hospital[mt_hospital_names]

        # From this reduced dataframe, pick out
        # the smallest time in each row and
        # the MT hospital that it belongs to.
        # Store the results in this DataFrame:
        df_nearest_mt = pd.DataFrame(index=df_time_inter_hospital.index)
        # The smallest time in each row:
        df_nearest_mt['transfer_unit_travel_time'] = (
            df_time_inter_hospital.min(axis='columns'))
        # The name of the column containing the smallest time in each row:
        df_nearest_mt['transfer_unit_postcode'] = (
            df_time_inter_hospital.idxmin(axis='columns'))

        # Update the feeder units list with anything specified
        # by the user.
        # df_services = self.national_hospital_services
        df_services_to_update = df_stroke_teams[
            df_stroke_teams['transfer_unit_postcode'] != 'nearest']
        units_to_update = df_services_to_update['postcode'].values
        transfer_units_to_update = df_services_to_update[
            'transfer_unit_postcode'].values
        for u, unit in units_to_update:
            transfer_unit = transfer_units_to_update[u]

            # Find the time to this MT unit.
            mt_time = df_time_inter_hospital.loc[unit][transfer_unit]

            # Update the chosen nearest MT unit name and time.
            df_nearest_mt.at[unit, 'transfer_unit_postcode'] = transfer_unit
            df_nearest_mt.at[unit, 'transfer_unit_travel_time'] = mt_time

        # Only keep units offering IVT.
        mask = df_nearest_mt.index.isin(ivt_hospital_names)
        df_nearest_mt = df_nearest_mt[mask]

        # # Store in self:
        # self.national_ivt_feeder_units = df_nearest_mt

        return df_nearest_mt

    # def _find_national_transfer_travel(self):
    #     """
    #     Data for the transfer stroke unit of each national stroke unit.

    #     Stores
    #     ------

    #     national_mt_transfer_time:
    #         dict. Each stroke unit's travel time to their nearest
    #         MT transfer unit.

    #     national_mt_transfer_unit:
    #         dict. Each stroke unit's nearest MT transfer unit's name.
    #     """
    #     # Load and parse inter hospital travel time for MT
    #     inter_hospital_time = self.national_ivt_feeder_units

    #     self.national_mt_transfer_time = dict(
    #         inter_hospital_time['time_nearest_mt'])
    #     self.national_mt_transfer_unit = dict(
    #         inter_hospital_time['name_nearest_mt'])
