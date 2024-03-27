"""
Generic functions for certain tasks such as saving and loading files.
"""
import numpy as np
import pandas as pd
from shapely import wkt  # for making geometry.
import geopandas


def find_multiindex_column_names(gdf, **kwargs):
    """
    Find the full column name to match a partial column name.

    Example usage:
    find_multiindex_column_name(gdf, scenario='any', property='geometry')

    Inputs
    ------
    gdf    - GeoDataFrame.
    kwargs - in format level_name=column_name for column level names
             in the gdf column MultiIndex.

    Returns
    -------
    cols - list or str or tuple. The column name(s) matching the
           requested names in those levels.
    """
    masks = [
        gdf.columns.get_level_values(level).isin(col_list)
        for level, col_list in kwargs.items()
    ]
    mask = np.all(masks, axis=0)
    cols = gdf.columns[mask]
    if len(cols) == 1:
        cols = cols.values[0]
    elif len(cols) == 0:
        cols = ''  # Should throw up a KeyError when used to index.
    return cols


def convert_df_to_gdf(df):
    """
    Convert a dataframe with a 'geometry' column to a GeoDataFrame.

    Useful for reading in a csv from file.

    Inputs
    ------
    df - pd.DataFrame.

    Returns
    -------
    gdf - GeoDataFrame. Same as input but with geometry column set.
    """
    df = df.copy()
    # Find the intended geometry column in the "property" column level:
    col = find_multiindex_column_names(df, property='geometry')
    try:
        gdf = geopandas.GeoDataFrame(
            df,
            geometry=col
            )
    except TypeError:
        # # Convert to a proper geometry column:
        df[col] = df[col].apply(wkt.loads)
        # Convert to a proper GeoDataFrame:
        gdf = geopandas.GeoDataFrame(
            df,
            geometry=col
            )
    return gdf
