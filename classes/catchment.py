"""
Catchment class to find LSOAs nearest given stroke units.

Given a dataframe of stroke units and which services they provide,
this class can find each unit's chosen transfer unit and each unit's
catchment area of LSOAs.
"""
import numpy as np
import pandas as pd
import yaml
from importlib_resources import files


class Catchment(object):
    """
    Global variables for model.

    class Catchment():

    Attributes
    ----------
    name:
        Name for this Catchment instance.

    catchment_yml:
        Name of a yml to load instance attributes from.

    limit_to_england:
        Whether to limit LSOA selection to England.

    limit_to_wales:
        Whether to limit LSOA selection to Wales.

    lsoa_catchment_type:
        Whether to only keep selected stroke units ('island') or
        to keep all units nationally for LSOA catchment calculation.

    Methods
    -------
    load_catchment_from_files:
        Load in attributes from a .yml.

    save_to_file:
        Save the variable dict as a .yml file.

    main:
        Calculate LSOA catchment for the selected units.

    get_unit_services:
        Load the stroke units data from file.

    get_transfer_units:
        Find each stroke unit's chosen transfer unit.

    find_national_mt_feeder_units:
        Find wheel-and-spoke IVT feeder units to each MT unit.

    calculate_lsoa_catchment:
        Calculate the LSOAs caught by each stroke unit.

    find_each_lsoa_chosen_unit:
        Extract LSOA unit data from LSOA-unit travel matrix.

    find_lsoa_catchment:
        Wrapper to load travel time matrix and pick out LSOA data.

    limit_lsoa_catchment_to_selected_units:
        Choose which LSOAs are selected using regions and units.
    """

    def __init__(
            self,
            *initial_data,
            **kwargs
            ):
        """
        Constructor method for model parameters

        """
        # ----- Directory setup -----
        # Name that will also be used for output directory:
        self.name = 'catchment'

        # ----- Geography -----
        # Which LSOAs, stroke units, and regions will we use?
        self.limit_to_england = False
        self.limit_to_wales = False
        self.lsoa_catchment_type = 'nearest'

        # ----- Overwrite default values -----
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # ----- Load Catchment from files -----
        if hasattr(self, 'catchment_yml'):
            self.load_catchment_from_files(self.catchment_yml)
        else:
            pass

    def load_catchment_from_files(self):
        """
        Load in attributes from a .yml.
        """
        # Import the kwargs from provided yml file:
        with open(self.catchment_yml, 'r') as f:
            catchment_vars_imported = yaml.safe_load(f)
        # Save the imported kwargs to self:
        for key, val in catchment_vars_imported.items():
            setattr(self, key, val)

    def save_to_file(
            self,
            file_yml='catchment_output.yml'
            ):
        """
        Save the variable dict as a .yml file.

        Inputs
        ------
        file_yml: str. Path to a yml file for saving.
        """
        catchment_vars = vars(self)

        # Only keep a selection of params:
        types_to_keep = [float, int, str]

        vars_to_save = {}
        for key, val in catchment_vars.items():
            if any([isinstance(val, t) for t in types_to_keep]):
                if isinstance(val, np.float64):
                    val = val.tolist()
                vars_to_save[key] = val

        with open(file_yml, 'w') as f:
            yaml.dump(vars_to_save, f)

    # ###############################
    # ##### MAIN SETUP FUNCTION #####
    # ###############################
    def main(self, units):
        """
        Calculate LSOA catchment for the selected units.

        Takes the input units and their services and defines
        each unit's transfer unit (df_transfer) and each
        LSOA's chosen unit (df_lsoa).

        Inputs
        ------
        units - pd.DataFrame. Which units are selected, offer
                IVT, MT, MSU etc. Must match the format and columns
                of the df returned by get_unit_services().

        Returns
        -------
        return_dict - dict. The main results dataframes containing info
                      on units, transfer units, and LSOA catchment.
        """
        self.df_units = units

        self.get_transfer_units()

        self.calculate_lsoa_catchment()

        return_dict = {
            'df_units': self.df_units,
            'df_transfer': self.df_transfer,
            'df_lsoa': self.df_lsoa,
            }
        return return_dict

    # #########################
    # ##### UNIT SERVICES #####
    # #########################
    def get_unit_services(self):
        """
        Load the stroke units data from file.

        Returns
        -------
        df - pd.DataFrame. Information about stroke units and their
             services.
        """
        # # Relative import from package files:
        path_to_file = files('stroke_maps.data').joinpath(
            'stroke_units_regions.csv')
        df = pd.read_csv(path_to_file)

        # Drop the LSOA column.
        df = df.drop('lsoa', axis='columns')
        # Add a "selected" column for user input.
        df['selected'] = 0

        df = df.set_index('postcode')

        return df

    # ##########################
    # ##### TRANSFER UNITS #####
    # ##########################
    def get_transfer_units(self):
        """
        Find each stroke unit's chosen transfer unit.

        Stores
        ------
        df_transfer - pd.DataFrame. Stores info on each unit's
                      transfer unit and whether the transfer should
                      be used for the selected units.
        """
        # Find which IVT units are feeders to each MT unit:
        transfer = self.find_national_mt_feeder_units(self.df_units)
        transfer = transfer.reset_index()
        # Index: none
        # Columns: 'postcode', 'transfer_unit_postcode',
        # 'transfer_unit_travel_time'

        units = self.df_units.copy()
        # Index: 'Postcode'
        # Columns: names, services, regions etc. ...
        units = units.reset_index()

        # Label selected stroke units.
        selected_units = units['postcode'][units['selected'] == 1].tolist()
        mask = (
            (transfer['postcode'].isin(selected_units)) &
            (transfer['transfer_unit_postcode'].notna())
        )
        transfer['selected'] = 0
        transfer.loc[mask, 'selected'] = 1
        transfer = transfer.set_index('postcode')

        self.df_transfer = transfer

    def find_national_mt_feeder_units(self, df_stroke_teams):
        """
        Find wheel-and-spoke IVT feeder units to each MT unit.

        Inputs
        ------
        df_stroke_teams - pd.DataFrame. Contains info on each unit
                          and the services it provides (IVT, MT, MSU).

        Returns
        ------
        df_nearest_mt - pd.DataFrame. Each row is a stroke unit.
                        Columns are its postcode, its transfer unit
                        postcode and travel time.
        """
        df_stroke_teams = df_stroke_teams.copy()
        df_stroke_teams = df_stroke_teams.reset_index()
        # Pick out the names of hospitals offering IVT:
        mask_ivt = (df_stroke_teams['use_ivt'] == 1)
        ivt_hospital_names = df_stroke_teams['postcode'][mask_ivt].values
        # Pick out the names of hospitals offering MT:
        mask_mt = (df_stroke_teams['use_mt'] == 1)
        mt_hospital_names = df_stroke_teams['postcode'][mask_mt].values

        # Only define transfer for units offering IVT.
        mask = df_stroke_teams['postcode'].isin(ivt_hospital_names)
        df_stroke_teams.loc[~mask, 'transfer_unit_postcode'] = 'none'

        # Firstly, determine MT feeder units based on travel time.
        # Each stroke unit will be assigned the MT unit that it is
        # closest to in travel time.
        # Travel time matrix between hospitals:
        # Relative import from package files:
        path_to_file = files('stroke_maps.data').joinpath(
            'inter_hospital_time_calibrated.csv')
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

        # Make sure the complete list of stroke teams is included:
        df_nearest_mt = df_nearest_mt.reset_index()
        df_nearest_mt = df_nearest_mt.rename(
            columns={'from_postcode': 'postcode'})
        df_nearest_mt = pd.merge(
            df_nearest_mt, df_stroke_teams['postcode'],
            on='postcode', how='right')
        df_nearest_mt = df_nearest_mt.set_index('postcode')

        # Update the feeder units list with anything specified
        # by the user.
        df_services_to_update = df_stroke_teams[
            df_stroke_teams['transfer_unit_postcode'] != 'nearest']
        units_to_update = df_services_to_update['postcode'].values
        transfer_units_to_update = df_services_to_update[
            'transfer_unit_postcode'].values
        for u, unit in enumerate(units_to_update):
            transfer_unit = transfer_units_to_update[u]
            if transfer_unit == 'none':
                # Set values to missing:
                transfer_unit = pd.NA
                mt_time = pd.NA
            else:
                # Find the time to this MT unit.
                mt_time = df_time_inter_hospital.loc[unit][transfer_unit]

            # Update the chosen nearest MT unit name and time.
            df_nearest_mt.at[unit, 'transfer_unit_postcode'] = transfer_unit
            df_nearest_mt.at[unit, 'transfer_unit_travel_time'] = mt_time

        return df_nearest_mt

    # #########################
    # ##### SELECTED LSOA #####
    # #########################
    def calculate_lsoa_catchment(self):
        """
        Calculate the LSOAs caught by each stroke unit.

        For the 'island' catchment type, nothing exists except the
        selected units and LSOAs in regions (English Integrated Care
        Boards ICBs and Welsh Local Health Boards LHBs) that contain
        selected units.
        For other catchment types, all catchment for all units is
        calculated.

        Stores
        ------
        df_lsoa - pd.DataFrame. Contains one row per LSOA and columns
                  for its selected unit and travel time.
        """
        units = self.df_units
        regions_selected = sorted(list(set(units.loc[
            units['selected'] == 1, 'region_code'])))
        units_selected = units.index[units['selected'] == 1].tolist()

        # Teams providing IVT:
        teams_with_services = units[units['use_ivt'] == 1].index.tolist()

        if self.lsoa_catchment_type == 'island':
            # Only use the selected stroke units:
            teams_to_limit = units_selected
            # Find list of selected regions:
            regions_to_limit = regions_selected
        else:
            teams_to_limit = []
            regions_to_limit = []

        # Only keep selected teams that offer IVT:
        teams_to_limit = list(set(teams_with_services + units_selected))

        # For all LSOA:
        df_catchment = self.find_lsoa_catchment(teams_to_limit)

        # Mark selected LSOA:
        df_catchment = self.limit_lsoa_catchment_to_selected_units(
            df_catchment,
            regions_to_limit=regions_to_limit,
            units_to_limit=units_selected,
            limit_to_england=self.limit_to_england,
            limit_to_wales=self.limit_to_wales
            )

        self.df_lsoa = df_catchment

    def find_each_lsoa_chosen_unit(self, df_time_lsoa_to_units):
        """
        Extract LSOA unit data from LSOA-unit travel matrix.

        Inputs
        ------
        df_time_lsoa_to_units - pd.DataFrame. Travel time matrix
                                with columns already limited as needed.

        Returns
        -------
        df_results - pd.DataFrame. One row per LSOA, columns for
                     chosen unit and travel time.
        """
        # Put the results in this dataframe where each row
        # is a different LSOA:
        df_results = pd.DataFrame(index=df_time_lsoa_to_units.index)
        # The smallest time in each row:
        df_results['unit_travel_time'] = (
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
        """
        Wrapper to load travel time matrix and pick out LSOA data.

        Inputs
        ------
        teams_to_limit - list. Only keep these units in the travel
                         matrix columns.

        Returns
        -------
        df_catchment - pd.DataFrame. One row per LSOA, columns for
                       chosen unit and travel time.
        """
        # Load travel time matrix:
        # # Relative import from package files:
        path_to_file = files('stroke_maps.data').joinpath(
            'lsoa_travel_time_matrix_calibrated.csv')
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
            regions_to_limit=[],
            units_to_limit=[],
            limit_to_england=False,
            limit_to_wales=False
            ):
        """
        Choose which LSOAs are selected using regions and units.

        Optionally limit the LSOAs to only a few regions.
        Optionally limit the LSOAs to only those caught by
        selected units.
        Optionally limit the LSOAs to only those in England or
        only those in Wales.

        Inputs
        ------
        df_catchment     - pd.DataFrame. LSOAs and their chosen units.
        regions_to_limit - list. List of regions to limit to.
        units_to_limit   - list. List of units to limit to.
        limit_to_england - bool. Whether to only keep English LSOA.
        limit_to_wales   - bool. Whether to only keep Welsh LSOA.

        Returns
        -------
        df_catchment - pd.DataFrame. The input dataframe with added
                       columns for LSOA codes and whether the LSOA
                       is selected.
        """
        # Load in all LSOA names, codes, regions...
        # Relative import from package files:
        path_to_file = files('stroke_maps.data').joinpath(
            'regions_lsoa_ew.csv')
        # Load and parse unit data:
        path_to_file = './data/regions_lsoa_ew.csv'
        df_lsoa = pd.read_csv(path_to_file)
        # Columns: [lsoa, lsoa_code, region_code, region, region_type,
        #           icb_code, icb, isdn]

        # Keep a copy of the original catchment columns for later:
        cols_df_catchment = df_catchment.columns.tolist()
        # Merge in region information to catchment:
        df_catchment.reset_index(inplace=True)
        df_catchment = pd.merge(
            df_catchment, df_lsoa,
            left_on='LSOA', right_on='lsoa', how='left'
        )
        df_catchment.drop('LSOA', axis='columns', inplace=True)
        df_catchment.set_index('lsoa', inplace=True)

        # Limit rows to LSOA in requested regions:
        if len(regions_to_limit) > 0:
            # Limit the results to only LSOAs in regions
            # containing selected units.
            mask = df_catchment['region_code'].isin(regions_to_limit)
            df_catchment = df_catchment.loc[mask].copy()
        elif len(units_to_limit) > 0:
            # Limit the results to only LSOAs that are caught
            # by selected units.
            mask = df_catchment['unit_postcode'].isin(units_to_limit)
        else:
            mask = [True] * len(df_catchment)

        df_catchment['selected'] = 0
        df_catchment.loc[mask, 'selected'] = 1

        # If requested, remove England or Wales.
        if limit_to_england:
            mask = df_lsoa['region_type'] == 'LHB'
            df_catchment['selected'][mask] = 0
        elif limit_to_wales:
            mask = df_lsoa['region_type'] == 'SICBL'
            df_catchment['selected'][mask] = 0

        # Restore the shortened catchment DataFrame to its starting columns
        # plus the useful regions:
        cols = cols_df_catchment + ['lsoa_code', 'selected']
        # ['region', 'region_code', 'region_type']
        df_catchment = df_catchment[cols]

        return df_catchment
