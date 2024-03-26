"""
Combine data from multiple sets of Catchment data.

Also a function to combine one set of Catchment data with another
data set with the same index.
"""
import numpy as np
import pandas as pd
from itertools import combinations


class Combine(object):
    """
    Combine data from multiple sets of Catchment data.

    class Combine():

    Attributes
    ----------
    (none)

    Methods
    -------
    combine_inputs_and_results:
        Wrapper for pd.merge().

    combine_selected_units:
        Combine units data.

    combine_selected_transfer:
        Combine transfer units data.

    combine_selected_lsoa:
        Combine LSOA data.

    _diff_data:
    Make new data of this column minus that column across scenarios.

    _hstack_multiple_dataframes:
        Combine multiple DataFrames from different scenarios into here.

    _merge_multiple_dataframes:
        Stack multiple DataFrames on top of the other, keep columns.

    """
    def __init__(self, *initial_data, **kwargs):

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

    # ####################
    # ##### WRAPPERS #####
    # ####################
    def combine_inputs_and_results(
            self,
            df_input,
            df_results,
            how='left'
            ):
        """
        Wrapper for pd.merge().

        Expect df_results to have column levels named
        'property' and 'subtype'.

        Must have matching index content.
        (pd.merge doesn't work well with MultiIndex merging and
        left_on, right_on).
        """
        # Target column heading level names:
        headers = df_results.columns.names

        # Create column levels for the input dataframe.
        # Create the correct total number of column names and leave
        # them blank.
        df_input_cols = [[''] * len(df_input.columns)] * len(headers)

        if isinstance(df_input.columns, pd.MultiIndex):
            # Already have multiple column levels.
            # Check names...
            if np.any(np.array(df_input.columns.names) == None):
                # Can't check the column header names.
                err = ''.join([
                    'Please set the column header names of df_input ',
                    'with `df_input.columns.names = headers`.'
                    ])
                raise KeyError(err) from None
            else:
                for header in np.array(df_input.columns.names):
                    # Find where this column level needs to be
                    # to match df_results.
                    ind = headers.index(header)
                    # Update the new column array with column names from
                    # the input dataframe.
                    df_input_cols[ind] = df_input.columns
        else:
            # Find where this column level needs to be
            # to match df_results.
            ind = headers.index('property')
            # Update the new column array with column names from
            # the input dataframe.
            df_input_cols[ind] = df_input.columns

        # Set up a new input DataFrame with the required column levels.
        df_input = pd.DataFrame(
            df_input.values,
            index=df_input.index,
            columns=df_input_cols
            )
        df_input.columns.names = headers

        # Now that the column header levels match, merge:
        df = pd.merge(df_input, df_results,
                      left_index=True, right_index=True, how=how)
        return df

    def combine_selected_units(
            self,
            dict_scenario_df_to_merge
            ):
        """
        Combine units data.

        Each DataFrame input:
        +------+-------------+-------------+
        |      |   time_1    |    shift_1  |    property
        +------+------+------+------+------+
        | Unit | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy |
        |    2 | x.xx | x.xx | y.yy | y.yy |
        |    3 | x.xx | x.xx | y.yy | y.yy |
        |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy |
        +------+------+------+------+------+

        Resulting DataFrame:
        +------+------+------+------+------+------+------+
        |  any |  scenario_1 |  scenario_2 |    diff     |    scenario
        +------+------+------+------+------+------+------+
        |      |    shift    |    shift    |    shift    |    property
        +------+------+------+------+------+------+------+
        | Unit | mean |  std | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    2 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    3 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |  ... |  ... |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        +------+------+------+------+------+------+------+

        Inputs
        ------
        dict_scenario_df_to_merge - dict of pd.DataFrame.
            Each key in the dict will be the 'scenario' column name
            for the matching data. Each DataFrame in the dict will
            be stacked side-by-side with the Unit index matching.

        Returns
        -------
        df - pd.DataFrame. The merged dataframe.
        """
        df = self._hstack_multiple_dataframes(
            dict_scenario_df_to_merge,
            cols_for_scenario=[
                'use_ivt',
                'use_mt',
                'use_msu',
                'selected',
                'transfer_unit_postcode',
                'time_ambulance_called',
                'time_ambulance_arrival',
                'time_ambulance_leaves_scene',
                'time_admitting_unit_arrival',
                'time_needle',
                'time_transfer_unit_arrival',
                'time_puncture',
                'mRS shift',
                'utility_shift',
                'mRS 0-2'
            ])

        # Create new columns of this diff that:
        cols_to_keep = ['utility_shift', 'mRS shift', 'mRS 0-2']
        df = self._diff_data(df, cols_to_keep)

        # Rename the MultiIndex column names:
        headers = ['scenario', 'property']
        headers_ref = list(
            dict_scenario_df_to_merge.values())[0].columns.names
        if 'subtype' in headers_ref:
            headers.append('subtype')
        df.columns = df.columns.set_names(headers)

        # Put 'property' above 'scenario':
        df = df.swaplevel('scenario', 'property', axis='columns')

        return df

    def combine_selected_transfer(
            self,
            dict_scenario_df_to_merge
            ):
        """
        Combine transfer units data.

        Each file input:
        +------+-----+---------------+-----------------+
        | Unit | ... | transfer_unit | transfer_coords |    property
        +------+-----+---------------+-----------------+
        |    1 | ... |             2 |  -x.xx, yy.yy   |
        |    2 | ... |             2 |  -x.xx, yy.yy   |
        |    3 | ... |             2 |  -x.xx, yy.yy   |
        |  ... | ... |           ... |       ...       |
        |    n | ... |             4 |  -x.xx, yy.yy   |
        +------+-----+---------------+-----------------+

        Resulting DataFrame:
                               +------------+------------+
                               | scenario_1 | scenario_2 |    scenario
        +------+---------------+------------+------------+
        | Unit | transfer_unit |        Use |        Use |    property
        +------+---------------+------------+------------+
        |    1 |             1 |          1 |          0 |
        |    2 |             1 |          1 |          0 |
        |    3 |             1 |          1 |          1 |
        |  ... |           ... |        ... |        ... |
        |    1 |             9 |          0 |          1 |
        +------+---------------+------------+------------+

        Inputs
        ------
        dict_scenario_df_to_merge - dict of pd.DataFrame.
            Each key in the dict will be the 'scenario' column name
            for the matching data. Each DataFrame in the dict will
            be stacked side-by-side with the Unit index matching.

        Returns
        -------
        df - pd.DataFrame. The merged dataframe.
        """
        # Merge the separate files based on combo of unit and
        # transfer unit, two indexes.
        df = self._hstack_multiple_dataframes(
            dict_scenario_df_to_merge,
            # add_use_column=True,
            cols_for_scenario=['selected', ],
            extra_cols_for_index=['transfer_unit_postcode']
            )

        # Rename the MultiIndex column names:
        df.columns = df.columns.set_names(['scenario', 'property'])

        # Put 'property' above 'scenario':
        df = df.swaplevel('scenario', 'property', axis='columns')

        return df

    def combine_selected_lsoa(
            self, dict_scenario_df_to_merge):
        """
        Combine LSOA data.

        Each file input:
        +------+-------------+-------------+
        |      |   time_1    |    shift_1  |    property
        +------+------+------+------+------+
        | LSOA | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy |
        |    2 | x.xx | x.xx | y.yy | y.yy |
        |    3 | x.xx | x.xx | y.yy | y.yy |
        |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy |
        +------+------+------+------+------+

        Resulting DataFrame:
        +------+------+------+------+------+------+------+
        |      |  scenario_1 |  scenario_2 |    diff     |    scenario
        +------+------+------+------+------+------+------+
        |      |    shift    |    shift    |    shift    |    property
        +------+------+------+------+------+------+------+
        | LSOA | mean |  std | mean |  std | mean |  std |    subtype
        +------+------+------+------+------+------+------+
        |    1 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    2 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |    3 | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        |  ... |  ... |  ... |  ... |  ... |  ... |  ... |
        |    n | x.xx | x.xx | y.yy | y.yy | z.zz | z.zz |
        +------+------+------+------+------+------+------+

        Inputs
        ------
        dict_scenario_df_to_merge - dict of pd.DataFrame.
            Each key in the dict will be the 'scenario' column name
            for the matching data. Each DataFrame in the dict will
            be stacked side-by-side with the Unit index matching.

        Returns
        -------
        df - pd.DataFrame. The merged dataframe.
        """
        df = self._hstack_multiple_dataframes(
            dict_scenario_df_to_merge,
            cols_for_scenario=':'
            )
        # TO DO - maybe change this so that all columns are combined always and at the end remove duplicate columns, move to 'any' scenario --------------

        # Create new columns of this diff that:
        cols_to_keep = ['utility_shift', 'mRS shift', 'mRS 0-2']
        df = self._diff_data(df, cols_to_keep)

        # Rename the MultiIndex column names:
        headers = ['scenario', 'property']
        h_ref = list(dict_scenario_df_to_merge.values())[0].columns.names
        if 'subtype' in h_ref:
            headers.append('subtype')
        df.columns = df.columns.set_names(headers)

        # Put 'property' above 'scenario':
        df = df.swaplevel('scenario', 'property', axis='columns')

        return df

    # #####################
    # ##### COMBINING #####
    # #####################
    def _diff_data(self, df, cols_to_diff):
        """
        Make new data of this column minus that column across scenarios.

        Inputs
        ------
        df           - pd.DataFrame. Contains the columns to diff.
        cols_to_diff - list. Column names to take the difference of
                       across two scenarios.
        """
        # Change to select top level of multiindex:
        scenario_name_list = sorted(list(set(
            df.columns.get_level_values(0).to_list())))
        try:
            # Drop 'any' scenario:
            scenario_name_list.remove('any')
        except ValueError:
            # no 'any' scenario here.
            pass

        # For scenarios [A, B, C], get a list of pairs
        # [[A, B], [A, C], [B, C]].
        scenario_name_pairs = list(combinations(scenario_name_list, 2))

        for c in cols_to_diff:
            # Take the difference between each pair of scenarios:
            for pair in scenario_name_pairs:
                p0 = pair[0]
                p1 = pair[1]
                diff_col_name = f'diff_{p0}_minus_{p1}'

                data0 = df[p0][c]
                data1 = df[p1][c]
                try:
                    for col in data0.columns:
                        if col in ['mean', 'median']:
                            # Take the difference between averages.
                            data_diff = data0[col] - data1[col]
                        elif col in ['std']:
                            # Propagate errors for std.
                            # Convert pandas NA to numpy NaN.
                            d0 = data0[col].copy().pow(2.0)
                            d1 = data1[col].copy().pow(2.0)
                            d2 = d0.add(d1, fill_value=0)
                            data_diff = d2.pow(0.5)
                        else:
                            # Don't know what to do with the rest yet. ----------------------
                            # TO DO
                            data_diff = ['help'] * len(df)
                        df[diff_col_name, c, col] = data_diff
                except AttributeError:
                    # No more nested column index levels.
                    data_diff = data0 - data1
                    df[diff_col_name, c] = data_diff
                    # TO DO - currently this just takes the difference
                    # as though it's a mean or a median. No way to
                    # propagate the error as though it's an std.
        return df

    def _hstack_multiple_dataframes(
            self,
            dict_scenario_df_to_merge,
            add_use_column=False,
            cols_for_scenario=[],
            extra_cols_for_index=[]
            ):
        """
        Combine multiple DataFrames from different scenarios into here.

        Stacks all DataFrames side by side with indexes matched.

        Each file input:
        +--------+-----------+----------+
        | Region | has_units | has_lsoa |
        +--------+-----------+----------+
        |      1 |      True |     True |
        |      2 |      True |     True |
        |      3 |     False |     True |
        |    ... |       ... |      ... |
        |      n |     False |     True |
        +--------+-----------+----------+

        Resulting DataFrame:
                 +----------------------+----------------------+
                 |      scenario_1      |      scenario_2      |
        +--------+-----------+----------+-----------+----------+
        | Region | has_units | has_lsoa | has_units | has_lsoa |
        +--------+-----------+----------+-----------+----------+
        |      1 |      True |     True |      True |     True |
        |      2 |      True |     True |     False |     True |
        |      3 |     False |     True |      True |     True |
        |    ... |       ... |      ... |       ... |      ... |
        |      n |     False |     True |     False |     True |
        +--------+-----------+----------+-----------+----------+

        Inputs
        ------
        dict_scenario_df_to_merge - dict of pd.DataFrame.
            Each key in the dict will be the 'scenario' column name
            for the matching data. Each DataFrame in the dict will
            be stacked side-by-side with the Unit index matching.
        add_use_column - bool. Whether to add a new column 'use'
            where 1 means that row is used in that scenario.
        cols_for_scenario - list. Column names that should be kept
            separately for each scenario.
        extra_cols_for_index - list. Set index to the initial index
            column and any columns in this list.

        Returns
        -------
        df - pd.DataFrame. The merged dataframe.
        """
        dfs_to_merge = {}

        for scenario_name, df in dict_scenario_df_to_merge.items():
            if len(extra_cols_for_index) > 0:
                iname = list(df.index.names)
                df = df.reset_index()
                df = df.set_index(iname + extra_cols_for_index)

            if len(dfs_to_merge.items()) < 1:
                shared_col_name = df.index.name

            if isinstance(cols_for_scenario, str):
                # Use all columns.
                pass
            else:
                split_bool = ((len(cols_for_scenario) != (len(df.columns))))
                split_for_any = True if split_bool else False
                if split_for_any:
                    # Find the names of these columns in this df.
                    # (so can specify one level of multiindex only).
                    scenario_cols = cols_for_scenario

                    if len(dfs_to_merge.items()) < 1:
                        # First time around this loop.
                        # Split off these columns from this scenario.
                        df_any = df.copy().drop(
                            cols_for_scenario, axis='columns')
                        dfs_to_merge['any'] = df_any
                    else:
                        pass

                    # Remove columns for "any" scenario:
                    df = df[scenario_cols]

            if add_use_column:
                shared_col_is_list = ((isinstance(shared_col_name, list)) |
                                      (isinstance(shared_col_name, tuple)))
                # TO DO ^ make this more generic.
                if shared_col_is_list:
                    use_col = tuple(
                        [b for b in shared_col_name[:-1]] + ['Use'])
                else:
                    use_col = 'Use'
                # Make a new column named use_col and set all values to 1:
                # (use "assign" to avoid annoying warning, value set on
                # copy of slice)
                df = df.assign(**{use_col: 1})
            else:
                pass

            dfs_to_merge[scenario_name] = df

        # Can't concat without index columns.
        data = pd.concat(
            dfs_to_merge.values(),
            axis='columns',
            keys=dfs_to_merge.keys()  # Names for extra index row
            )

        try:
            # Move 'any' index to the far left:
            cols = list(np.unique(data.columns.get_level_values(0).to_list()))
            cols.remove('any')
            cols = ['any'] + cols
            data = data[cols]
        except ValueError:
            # No 'any' columns yet.
            pass

        # Sort rows by contents of index:
        data = data.sort_index()

        # Did have dtype float/str from missing values, now want int
        # if possible:
        data = data.convert_dtypes()

        return data

    def _merge_multiple_dataframes(
            self,
            dict_scenario_df_to_merge,
            merge_col='lsoa_code'
            ):
        """
        Stack multiple DataFrames on top of the other, keep columns.

        Inputs
        ------
        dict_scenario_df_to_merge - dict of pd.DataFrame.
            Each key in the dict will be the 'scenario' column name
            for the matching data. Each DataFrame in the dict will
            be stacked side-by-side with the Unit index matching.
        merge_col - str. Name of the column to match dfs on.

        Returns
        -------
        data - pd.DataFrame. The merged dataframe.
        """
        data = pd.DataFrame(columns=[merge_col])
        scenario_cols_list = []
        scenario_series_list = []

        for scenario_name, df in dict_scenario_df_to_merge.items():
            # Create a name for this scenario:
            scenario_series = pd.Series(
                [1]*len(df),
                index=df[merge_col],
                name=scenario_name
            )

            scenario_cols_list.append(scenario_name)
            scenario_series_list.append(scenario_series)
            # Add all new LSOAs to the bottom of the existing DataFrame
            # and remove any duplicate rows.
            data = pd.concat([data, df], axis=0).drop_duplicates()
        # Merge in the Series data:
        for s in scenario_series_list:
            data = pd.merge(
                data, s.reset_index(),
                left_on=merge_col, right_on=merge_col, how='left'
                )
        # Replace missing values with 0 in the scenario columns:
        data = data.fillna(
            value=dict(zip(scenario_cols_list,
                           [0]*len(scenario_cols_list))))
        data = data.convert_dtypes()
        # Sort rows:
        data = data.sort_values(merge_col)
        return data
