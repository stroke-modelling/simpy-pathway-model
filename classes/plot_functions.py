"""
Functions for drawing maps.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For blank patches.
import matplotlib.path as mpath        # For custom patches.
# For adjusting size of text scatter markers:
from PIL import ImageFont
from matplotlib import font_manager

import numpy as np


# ###################
# ##### HELPERS #####
# ###################
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


def combine_legend_sections(
        section_labels,
        handles_lists,
        labels_lists
        ):
    """
    Combine elements of multiple separate legends into one legend.

    +-----------+
    | o  Thing1 |   <-- Handles are the markers on the left,
    | x  Thing2 |       labels are the strings on the right.
    | *  Thing3 |
    +-----------+

    This combines multiple sets of handles and labels into one long
    set with new section labels (with invisible handles) between the
    sets.

    Inputs
    ------
    section_labels - list of str. Title for each section of legend.
    handles_lists  - list of lists. Patches for each legend.
    labels_lists   - list of lists. Labels for each legend.

    Returns
    -------
    handles_all - list. Patches for combined legend.
    labels_all - list. Labels for combined legend.
    """
    # Place the results in here:
    handles_all = []
    labels_all = []
    for i, section_label in enumerate(section_labels):
        # Add a blank handle to start of this handles list...
        blank_section_patch = mpatches.Patch(visible=False)
        handles_s = [blank_section_patch] + handles_lists[i]
        # ... and the section label to the start of this labels list.
        labels_s = [section_label] + labels_lists[i]
        # Add these lists to the combined list.
        handles_all += handles_s
        labels_all += labels_s
    return handles_all, labels_all


def set_legend_section_labels_to_bold(
        leg,
        section_labels,
        all_labels
        ):
    """
    Change the text of some legend entries to bold.

    Inputs
    ------
    leg            - plt.legend object.
    section_labels - list. Strings to become bold.
    all_labels     - list. All labels in the legend.
    """
    # Get the list of labels in the legend:
    leg1_list = leg.get_texts()
    for s in section_labels:
        # Where is the legend label that matches this section label?
        i = all_labels.index(s)
        # Update the label with the same index:
        leg1_list[i] = leg1_list[i].set_weight('heavy')
    return leg


def create_units_legend(
        ax,
        handles_lists,
        labels_lists,
        section_labels,
        **legend_kwargs
        ):
    """
    Create a legend for the unit markers with labels overlaid.

    Each legend handle will end up as a normal scatter marker
    with a string displayed on top. Looks like e.g. the letters
    'EX' contained in an oval.

    +-----------+
    | o  Thing1 |   <-- Handles are the markers on the left,
    | x  Thing2 |       labels are the strings on the right.
    | *  Thing3 |
    +-----------+

    Inputs
    ------
    handles_lists   - list of lists. One list of handles for each
                      legend.
    labels_lists    - list of lists. One list of labels for each
                      legend.
    section_labels  - list of str. Headings to separate the legends
                      after combination.
    **legend_kwargs - dict. Kwargs for plt.legend().

    Returns
    -------
    leg2 - plt.legend() object.
    """
    # By default, each handles list is in the format:
    # [[h[0], h[1], ... h[n]], [h2[0], h2[1], ... h2[n]]]
    # Where e.g. h is a list of the oval scatter markers
    # and h2 is a list of the string labels to place on top.
    # To combine the handle types, the list needs to be like:
    # [(h[0], h2[0]), (h[1], h2[1]), ...]
    # and it has to be pairs of tuple, not of list.
    # Rejig the list in the following loop:
    handles_list = []
    for hlist in handles_lists:
        hlist = np.array(hlist).T.tolist()
        hlist = [tuple(h) for h in hlist]
        handles_list.append(hlist)

    # Add a blank handle and a section label:
    handles_u, labels_u = combine_legend_sections(
        section_labels,
        handles_list,
        labels_lists
        )
    # Create the legend from the lists:
    leg2 = ax.add_artist(plt.legend(
        handles_u, labels_u, **legend_kwargs
        ))
    leg2 = set_legend_section_labels_to_bold(
        leg2, section_labels, labels_u)
    return leg2


# ####################
# ##### PLOTTING #####
# ####################
def draw_boundaries_by_contents(
        ax,
        gdf_boundaries_regions,
        kwargs_with_nowt={},
        kwargs_with_periphery={},
        kwargs_with_unit={}
        ):
    """
    Draw region boundaries formatted by contains unit or periphery bits.

    Inputs
    ------
    ax                     - plt.subplot().
    gdf_boundaries_regions - GeoDataFrame.
    kwargs_with_nowt       - dict. Format for regions with
                             nothing of interest.
    kwargs_with_periphery  - dict. Format for regions with
                             periphery bits.
    kwargs_with_unit       - dict. Format for regions with
                             selected unit.

    Returns
    -------
    ax - plt.subplot(). Same as input with the boundaries drawn on.
    """
    # Set up kwargs.
    kwargs_nowt = {
        'edgecolor': 'none',
        'facecolor': 'none',
    }
    kwargs_periphery = {
        'edgecolor': 'silver',
        'facecolor': 'none',
        'linewidth': 0.5,
        'linestyle': '--'
    }
    kwargs_unit = {
        'edgecolor': 'k',
        'facecolor': 'none',
        'linewidth': 0.5
    }
    # Update these with anything from the input dicts:
    kwargs_nowt = kwargs_nowt | kwargs_with_nowt
    kwargs_periphery = kwargs_periphery | kwargs_with_periphery
    kwargs_unit = kwargs_unit | kwargs_with_unit

    # Regions containing neither LSOAs nor stroke units here:
    try:
        mask = (
            (gdf_boundaries_regions['contains_unit'] == 0) &
            (gdf_boundaries_regions['contains_periphery_lsoa'] == 0) &
            (gdf_boundaries_regions['contains_periphery_unit'] == 0)
            )
    except KeyError:
        # Assume the LSOA column doesn't exist.
        mask = (
            (gdf_boundaries_regions['contains_unit'] == 0)
        )
    gdf_boundaries_with_nowt = gdf_boundaries_regions.loc[mask]
    if len(gdf_boundaries_with_nowt) > 0:
        ax = draw_boundaries(
            ax, gdf_boundaries_with_nowt,
            **kwargs_nowt
            )
        # Make these invisible but draw them anyway to make sure the
        # extent of the map is similar to other runs.
    else:
        pass

    # Regions containing LSOAs but not selected stroke units:
    # For now, treat extra regions the same regardless of whether
    # they contain an extra stroke unit or catch the LSOA.
    try:
        mask = (
            (gdf_boundaries_regions['contains_unit'] == 0) &
            ((gdf_boundaries_regions['contains_periphery_lsoa'] == 1) |
             (gdf_boundaries_regions['contains_periphery_unit'] == 1))
        )
    except KeyError:
        # Try again without the unit-catching column.
        try:
            mask = (
                (gdf_boundaries_regions['contains_unit'] == 0) &
                (gdf_boundaries_regions['contains_periphery_lsoa'] == 1)
            )
        except KeyError:
            # Assume the LSOA column doesn't exist. Don't plot this.
            mask = [False] * len(gdf_boundaries_regions)

    gdf_boundaries_with_lsoa = gdf_boundaries_regions.loc[mask]
    if len(gdf_boundaries_with_lsoa) > 0:
        ax = draw_boundaries(
            ax, gdf_boundaries_with_lsoa,
            **kwargs_periphery
            )
    else:
        pass

    # Regions containing selected stroke units:
    mask = (gdf_boundaries_regions['contains_unit'] == 1)
    gdf_boundaries_with_units = gdf_boundaries_regions.loc[mask]
    if len(gdf_boundaries_with_units) > 0:
        ax = draw_boundaries(
            ax, gdf_boundaries_with_units,
            **kwargs_unit
            )
    else:
        pass
    return ax


def draw_boundaries(ax, gdf, **kwargs):
    """
    Draw regions from a GeoDataFrame.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the regions.
    gdf    - GeoDataFrame. Stores geometry to be plotted.
    kwargs - dict. Keyword arguments to pass to plt.plot().

    Returns
    -------
    ax - pyplot axis. Same as input but with regions drawn on.
    """
    # Draw the main map with colours (choropleth):
    gdf.plot(
        ax=ax,              # Set which axes to use for plot (only one here)
        antialiased=False,  # Avoids artifact boundry lines
        **kwargs
        )
    return ax


def scatter_units(
        ax,
        gdf,
        return_handle=False,
        **kwargs
        ):
    """
    Draw scatter markers for IVT stroke units.

    Inputs
    ------
    ax            - pyplot axis. Where to draw the scatter markers.
    gdf           - GeoDataFrame. Stores stroke unit coordinates
                    and services.
    return_handle - bool. Whether to return the scatter marker
                    handles for making a legend later.
    kwargs        - dict. kwargs for plt.scatter().

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    kwargs_dict = dict(
        edgecolor='none',
        facecolor='LightCoral',
        linewidth=0.5,
        s=50,
        marker='o',
        zorder=2
    )

    # Overwrite default dict with user inputs:
    for key, val in kwargs.items():
        kwargs_dict[key] = val

    # Keep a copy of some defaults:
    default_marker_size = kwargs_dict['s']
    default_marker = kwargs_dict['marker']
    # facecolour = kwargs_dict['facecolor']

    # Create an ellipse marker by taking the standard circle
    # and stretching the x-coordinates.
    circle = mpath.Path.unit_circle()
    verts = np.copy(circle.vertices)
    verts[:, 0] *= 2.0
    ellipse = mpath.Path(verts, circle.codes)

    # Create a star marker by taking the standard star and pushing
    # out all coordinates away from the centre.
    star = mpath.Path.unit_regular_star(numVertices=5)
    verts = np.copy(star.vertices)
    radii = np.sqrt(verts[:, 0]**2.0 + verts[:, 1]**2.0)
    # Radii are either 0.5 (inner corner) or 1.0 (outer point).
    scale = 0.7 * (1.0 / radii)
    scale = np.sqrt(scale)
    verts[:, 0] = verts[:, 0] * scale
    verts[:, 1] = verts[:, 1] * scale
    # Also squash the y-axis down a bit:
    verts[:, 1] = verts[:, 1] * 0.75
    star_squash = mpath.Path(verts, star.codes)

    if return_handle:
        # Draw each point separately for use with the legend later.

        # Need a separate call for each when there is
        # an array of marker shapes.
        handles = []
        for row in gdf.index:
            gdf_m = gdf.loc[[row]]
            col_geometry = find_multiindex_column_names(
                gdf_m, property=['geometry'])
            # Update marker shape and size:
            try:
                col_marker = find_multiindex_column_names(
                    gdf_m, property=['marker'])
                marker = gdf_m[col_marker].values[0]
                kwargs_dict['marker'] = marker
            except KeyError:
                kwargs_dict['s'] = default_marker_size
                kwargs_dict['marker'] = default_marker

            if kwargs_dict['marker'] == '*':
                # Make the star bigger.
                kwargs_dict['s'] = 150
                kwargs_dict['marker'] = star_squash
            elif kwargs_dict['marker'] == 'o':
                kwargs_dict['s'] = 150
                kwargs_dict['marker'] = ellipse

            handle = ax.scatter(
                gdf_m[col_geometry].x,
                gdf_m[col_geometry].y,
                **kwargs_dict
                )
            handles.append(handle)
        return ax, handles
    else:

        col_geometry = find_multiindex_column_names(
            gdf, property=['geometry'])
        # Draw all points in one call.
        ax.scatter(
            gdf[col_geometry].x,
            gdf[col_geometry].y,
            **kwargs_dict
            )
        return ax


def plot_lines_between_units(ax, gdf, **line_kwargs):
    """
    Draw lines from stroke units to their MT transfer units.

    Inputs
    ------
    ax          - pyplot axis. Where to draw the scatter markers.
    gdf         - GeoDataFrame. Stores LineString objects that connect
                  each stroke unit to its MT transfer unit.
    line_kwargs - dict. Kwargs for plt.plot().

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    kwargs_dict = dict(
        edgecolor='k',
        linestyle='-',
        linewidth=2,
        zorder=1  # Place it beneath the scatter markers.
    )

    # Overwrite default dict with user inputs:
    for key, val in line_kwargs.items():
        kwargs_dict[key] = val

    # Draw a line connecting each unit to its MT unit.
    mask = gdf['selected'] == 1
    lines = gdf[mask]

    if 'colour_lines' in lines.columns:
        edgecolour = kwargs_dict['edgecolor']
        # Different colour for each line.
        for row in lines.index:
            gdf_m = lines.loc[[row]]
            if len(gdf_m['colour_lines'].values[0]) < 7:
                # Not in the format #rrggbb or #rrggbbaa.
                # Use input default value.
                colour = edgecolour
            else:
                colour = gdf_m['colour_lines']
            kwargs_dict['edgecolor'] = colour
            # Plot this single line as usual:
            gdf_m.plot(
                ax=ax,
                **kwargs_dict
            )
    else:
        lines.plot(
            ax=ax,
            **kwargs_dict
        )
    return ax


def draw_labels_short(
        ax,
        points,
        map_labels,
        leg_labels,
        colours=[],
        **kwargs
        ):
    """
    Draw labels from the geodataframe.

    Inputs
    ------
    ax         - plt.subplot().
    points     - pd.Series. Geometry of Points.
    map_labels - pd.Series. Text for the scatter markers.
    leg_labels - pd.Series. Text for legend entries.
    colours    - pd.Series. #rrggbbaa strings for label colour.
    **kwargs   - dict. kwargs for plt.scatter().

    Returns
    -------
    ax                 - plt.subplot(). Input axis with labels drawn on.
    markers_for_legend - List of handles for making a legend.
    labels_for_legend  - List of labels for the legend.
    """
    marker_kwargs = dict(
        s=8,
        edgecolor='none',  # to prevent border around text
        color='k'
    )
    # Update this with anything from the input dict:
    marker_kwargs = marker_kwargs | kwargs

    ref_s = marker_kwargs['s']

    font = font_manager.FontProperties()
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, marker_kwargs['s'])

    markers_for_legend = []
    labels_for_legend = []

    if len(colours) == 0:
        colours = [marker_kwargs['color']] * len(points)

    # Define "z" to shorten following "for" line:
    z = zip(
        points.x,
        points.y,
        map_labels,
        leg_labels,
        colours
        )
    for x, y, map_label, leg_label, colour in z:
        # Adjust label size based on its width.
        # Only need the ratio of height to width,
        # so find that for the same font outside the plot:
        left, top, right, bottom = font.getbbox(map_label)
        ref_height = bottom - top  # Yes really
        ref_width = right - left
        if ref_height >= ref_width:
            # If the label is taller than it is wide,
            # just use the normal marker size:
            s = ref_s
        else:
            # The label is too wide.
            # Using the reference s will mean the text is shrunk.
            # Scale up the marker size so that its height matches
            # the reference marker height
            # (squared because s is an area):
            s = ref_s * (ref_width / ref_height)**2.0
        # Update the kwargs with this marker size:
        marker_kwargs['s'] = s
        marker_kwargs['color'] = colour

        m = ax.scatter(
            x, y,
            marker=r'$\mathdefault{' + f'{map_label}' + '}$',
            zorder=5,
            **marker_kwargs
        )
        markers_for_legend.append(m)
        labels_for_legend.append(leg_label)
    return ax, markers_for_legend, labels_for_legend


def plot_dummy_axis(fig, ax, leg, side='left'):
    """
    Add dummy invisible axis to the side of an existing axis.

    So that extra_artists are not cut off when plt.show() crops.

    For now this assumes that the new axis should be anchored to
    "upper right" for new axis on left-hand-side and
    "upper left" for new axis on right-hand-side.
    For other options, make new if/else settings below.

    Inputs
    ------
    fig  - plt.figure().
    ax   - plt.subplot().
    leg  - plt.legend().
    side - str. Whether to place legend on the 'left' or 'right'.

    Returns
    -------
    fig - plt.figure(). Input fig extended to make room for legend.
    """
    # Current axis:
    abox = ax.get_window_extent().transformed(
        ax.figure.transFigure.inverted())
    x0 = abox.x0
    y0 = abox.y0
    width = abox.width
    height = abox.height

    # The size of the legend or other thing to make a dummy of:
    tbox = leg.get_window_extent().transformed(
        ax.figure.transFigure.inverted())

    new_width = tbox.width
    new_height = tbox.height
    if side == 'left':
        # Left-hand-side:
        new_x0 = x0 - tbox.width
        new_y0 = (y0 + height) - (tbox.height)
    else:
        # Right-hand-side:
        new_x0 = x0 + width
        new_y0 = (y0 + height) - (tbox.height)

    # Axes: [x0, y0, width, height].
    c = fig.add_axes([new_x0, new_y0, new_width, new_height])
    c.set_axis_off()
    return fig


# ######################
# ##### MAIN PLOTS #####
# ######################
def plot_map_selected_regions(
        gdf_boundaries_regions,
        gdf_points_units,
        ax=None,
        map_extent=[]
        ):
    """
    Make map of the selected regions and any units.

    Inputs
    ------
    gdf_boundaries_regions - GeoDataFrame.
    gdf_points_units       - GeoDataFrame.
    ax                     - plt.subplot().
    map_extent             - list. Axis limits [xmin, xmax, ymin, ymax].

    Returns
    -------
    ax            - plt.subplot(). Input ax with bits drawn on.
    extra_artists - list. Extra legends drawn on with plt.add_artist().
    """

    # Set up kwargs
    kwargs_with_unit = {
        'edgecolor': 'DimGray',
        'linewidth': 0.5,
        'facecolor': 'Gainsboro'
    }
    kwargs_with_nowt = {
        'edgecolor': 'silver',
        'linewidth': 0.5,
        'facecolor': 'WhiteSmoke'
        }

    label_size_units = 15
    label_size_regions = 30

    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax = draw_boundaries_by_contents(
        ax,
        gdf_boundaries_regions,
        kwargs_with_nowt=kwargs_with_nowt,
        kwargs_with_periphery=kwargs_with_nowt,
        kwargs_with_unit=kwargs_with_unit,
        )

    # In selected regions:
    col_selected = find_multiindex_column_names(
        gdf_points_units, property=['selected'])
    mask = gdf_points_units[col_selected] == 1
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        return_handle=True,
        edgecolor='k',
        )
    # Outside selected regions:
    mask = ~mask
    ax, handles_scatter_uns = scatter_units(
        ax,
        gdf_points_units[mask],
        facecolor='Pink',
        return_handle=True,
        edgecolor='DimGray',
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units[col_selected] == 1
    col_geometry = find_multiindex_column_names(
        gdf_points_units, property=['geometry'])
    col_short_code = find_multiindex_column_names(
        gdf_points_units, property=['short_code'])
    col_stroke_team = find_multiindex_column_names(
        gdf_points_units, property=['stroke_team'])
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units.loc[mask, col_geometry],
        gdf_points_units.loc[mask, col_short_code],
        gdf_points_units.loc[mask, col_stroke_team],
        s=label_size_units,
        color='k',
        # bbox=dict(facecolor='WhiteSmoke', edgecolor='r'),
    )
    # Outside selected regions:
    mask = ~mask
    ax, handles_uns, labels_uns = draw_labels_short(
        ax,
        gdf_points_units.loc[mask, col_geometry],
        gdf_points_units.loc[mask, col_short_code],
        gdf_points_units.loc[mask, col_stroke_team],
        s=label_size_units,
        color='DimGray',
    )

    # Label regions:
    # Change geometry so that we can use the point coordinates
    # in the following label function:
    gdf_boundaries_regions = gdf_boundaries_regions.set_geometry('point_label')
    # In selected regions:
    mask = gdf_boundaries_regions['contains_unit'] == 1
    ax, handles_rs, labels_rs = draw_labels_short(
        ax,
        gdf_boundaries_regions[mask].point_label,
        gdf_boundaries_regions[mask].short_code,
        gdf_boundaries_regions[mask].region,
        # weight='bold',
        s=label_size_regions,
        color='k'
    )
    # Outside selected regions:
    mask = ~mask
    ax, handles_rns, labels_rns = draw_labels_short(
        ax,
        gdf_boundaries_regions[mask].point_label,
        gdf_boundaries_regions[mask].short_code,
        gdf_boundaries_regions[mask].region,
        # weight='bold',
        s=label_size_regions,
        color='DimGray'
    )

    # Add legends.

    # Regions:
    # Add a blank handle and a section label:
    section_labels = ['Regions with selected units' + ' ' * 60 + '.',
                      'Other regions']
    handles_r, labels_r = combine_legend_sections(
        section_labels,
        [handles_rs, handles_rns],
        [labels_rs, labels_rns]
        )
    # Create the legend from the lists:
    leg1 = ax.add_artist(plt.legend(
        handles_r, labels_r, fontsize=6,
        bbox_to_anchor=[1.0, 1.0],
        loc='upper left'
        ))
    leg1 = set_legend_section_labels_to_bold(
        leg1, section_labels, labels_r)

    # Units:
    if len(labels_uns) > 0:
        section_labels = ['Selected units' + ' ' * 70 + '.',
                          'Other units']
        handles_lists = [
            [handles_scatter_us, handles_us],
            [handles_scatter_uns, handles_uns]
        ]
        labels_lists = [labels_us, labels_uns]
    else:
        section_labels = ['Selected units' + ' ' * 70 + '.']
        handles_lists = [
            [handles_scatter_us, handles_us]
        ]
        labels_lists = [labels_us]

    leg2 = create_units_legend(
        ax,
        handles_lists,
        labels_lists,
        section_labels,
        fontsize=6,
        bbox_to_anchor=[0.0, 1.0],
        loc='upper right'
        )

    if len(map_extent) > 0:
        # Limit to given extent:
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
    else:
        # Use default axis limits.
        pass

    ax.set_axis_off()  # Turn off axis line and numbers

    # Return extra artists so that bbox_inches='tight' line
    # in savefig() doesn't cut off the legends.
    # Adding legends with ax.add_artist() means that the
    # bbox_inches='tight' line ignores them.
    extra_artists = (leg1, leg2)

    return ax, extra_artists


def plot_map_selected_units(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_lines_transfer,
        ax=None,
        map_extent=[],
        ):
    """
    Make map of the selected units and the regions containing them.

    Inputs
    ------
    gdf_boundaries_regions - GeoDataFrame.
    gdf_points_units       - GeoDataFrame.
    gdf_lines_transfer     - GeoDataFrame.
    ax                     - plt.subplot().
    map_extent             - list. Axis limits [xmin, xmax, ymin, ymax].

    Returns
    -------
    ax            - plt.subplot(). Input ax with bits drawn on.
    extra_artists - list. Extra legends drawn on with plt.add_artist().
    """
    # Set up kwargs.
    # Region boundaries:
    kwargs_region_with_unit = {
        'edgecolor': 'DimGray',
        'linewidth': 0.5,
        'facecolor': 'Gainsboro'
        }
    kwargs_region_with_nowt = {
        'edgecolor': 'silver',
        'linewidth': 0.5,
        'facecolor': 'WhiteSmoke'
        }
    label_size_units = 15

    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax = draw_boundaries_by_contents(
        ax,
        gdf_boundaries_regions,
        kwargs_with_nowt=kwargs_region_with_nowt,
        kwargs_with_periphery=kwargs_region_with_nowt,
        kwargs_with_unit=kwargs_region_with_unit,
        # kwargs_selected=kwargs_selected,
        # kwargs_not_selected=kwargs_not_selected
        )

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'o')
    # Update MT units:
    col_use_mt = find_multiindex_column_names(
        gdf_points_units, property=['use_mt'])
    mask_mt = (gdf_points_units[col_use_mt] == 1)
    markers[mask_mt] = '*'
    # Store in the DataFrame:
    gdf_points_units['marker'] = markers

    # In selected regions:
    col_selected = find_multiindex_column_names(
        gdf_points_units, property=['selected'])
    mask = gdf_points_units[col_selected] == 1
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        return_handle=True,
        facecolor='WhiteSmoke',
        edgecolor='k'
        )
    # Outside selected regions:
    mask = gdf_points_units[col_selected] == 0
    ax, handles_scatter_uns = scatter_units(
        ax,
        gdf_points_units[mask],
        facecolor='WhiteSmoke',
        edgecolor='DimGray',
        return_handle=True
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units[col_selected] == 1
    col_geometry = find_multiindex_column_names(
        gdf_points_units, property=['geometry'])
    col_short_code = find_multiindex_column_names(
        gdf_points_units, property=['short_code'])
    col_stroke_team = find_multiindex_column_names(
        gdf_points_units, property=['stroke_team'])
    col_colour_lines = find_multiindex_column_names(
        gdf_points_units, property=['colour_lines'])
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units.loc[mask, col_geometry],
        gdf_points_units.loc[mask, col_short_code],
        gdf_points_units.loc[mask, col_stroke_team],
        s=label_size_units,
        colours=gdf_points_units.loc[mask, col_colour_lines],
        # color='k',
        # bbox=dict(facecolor='WhiteSmoke', edgecolor='r'),
    )
    # Outside selected regions:
    mask = ~mask
    ax, handles_uns, labels_uns = draw_labels_short(
        ax,
        gdf_points_units.loc[mask, col_geometry],
        gdf_points_units.loc[mask, col_short_code],
        gdf_points_units.loc[mask, col_stroke_team],
        s=label_size_units,
        color='DimGray',
        # bbox=dict(facecolor='GhostWhite', edgecolor='DimGray'),
    )

    # Units:
    if len(labels_uns) > 0:
        section_labels = ['Selected units' + ' ' * 70 + '.', 'Other units']
        handles_lists = [
            [handles_scatter_us, handles_us],
            [handles_scatter_uns, handles_uns]
        ]
        labels_lists = [labels_us, labels_uns]
    else:
        section_labels = ['Selected units' + ' ' * 70 + '.']
        handles_lists = [
            [handles_scatter_us, handles_us]
        ]
        labels_lists = [labels_us]

    leg = create_units_legend(
        ax,
        handles_lists,
        labels_lists,
        section_labels,
        fontsize=6,
        bbox_to_anchor=[0.0, 1.0],
        loc='upper right'
        )

    ax = plot_lines_between_units(
        ax,
        gdf_lines_transfer,
        edgecolor='LightCoral'
        )

    if len(map_extent) > 0:
        # Limit to given extent:
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
    else:
        # Use default axis limits.
        pass

    ax.set_axis_off()  # Turn off axis line and numbers

    extra_artists = (leg, )  # Has to be a tuple.

    return ax, extra_artists


def plot_map_catchment(
        gdf_boundaries_catchment,
        gdf_boundaries_regions,  # TO DO - make this optional...?
        gdf_points_units,
        gdf_lines_transfer,
        ax=None,
        map_extent=[],
        boundary_kwargs={},
        boundary_periphery_kwargs={}
        # catchment_type=''
        ):
    """
    Map the selected units, containing regions, and catchment areas.

    Inputs
    ------
    gdf_boundaries_catchment  - GeoDataFrame.
    gdf_boundaries_regions    - GeoDataFrame.
    gdf_points_units          - GeoDataFrame.
    gdf_lines_transfer        - GeoDataframe.
    ax                        - plt.subplot().
    map_extent                - list. Axis limits [xmin, xmax, ymin, ymax].
    boundary_kwargs           - dict. Kwargs for plt.plot() for
                                catchment areas.
    boundary_periphery_kwargs - dict. Kwargs for plt.plot() for
                                periphery catchment areas.

    Returns
    -------
    ax            - plt.subplot(). Input ax with bits drawn on.
    extra_artists - list. Extra legends drawn on with plt.add_artist().
    """
    label_size_units = 15

    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(12, 8))

    # LSOAs in selected units catchment:
    col_selected = find_multiindex_column_names(
        gdf_boundaries_catchment, property=['selected'])
    col_periphery_unit = find_multiindex_column_names(
        gdf_boundaries_catchment, property=['periphery_unit'])
    col_colour = find_multiindex_column_names(
        gdf_boundaries_catchment, property=['colour'])
    mask = (gdf_boundaries_catchment[col_selected] == 1)
    ax = draw_boundaries(
        ax,
        gdf_boundaries_catchment.loc[mask],
        color=gdf_boundaries_catchment.loc[mask, col_colour],
        **boundary_kwargs
        )

    # LSOAs in periphery units catchment:
    mask = (
        (gdf_boundaries_catchment[col_selected] == 0) &
        (gdf_boundaries_catchment[col_periphery_unit] == 1)
    )
    col_colour_periphery = find_multiindex_column_names(
        gdf_boundaries_catchment, property=['colour_periphery'])
    ax = draw_boundaries(
        ax,
        gdf_boundaries_catchment[mask],
        color=gdf_boundaries_catchment.loc[mask, col_colour_periphery],
        **boundary_periphery_kwargs
        )

    # Region boundaries:
    kwargs_region_with_unit = {
        'edgecolor': 'k',
        'linewidth': 0.5,
        'facecolor': 'none'
        }
    kwargs_region_with_lsoa = {
        'edgecolor': 'silver',
        'linewidth': 0.5,
        'facecolor': 'none'
        }
    kwargs_region_with_nowt = {
        'edgecolor': 'none',
        'facecolor': 'none'
        }

    ax = draw_boundaries_by_contents(
        ax,
        gdf_boundaries_regions,
        kwargs_with_nowt=kwargs_region_with_nowt,
        kwargs_with_periphery=kwargs_region_with_lsoa,
        kwargs_with_unit=kwargs_region_with_unit,
        )

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'o')
    # Update MT units:
    col_use_mt = find_multiindex_column_names(
        gdf_points_units, property=['use_mt'])
    mask_mt = (gdf_points_units[col_use_mt] == 1)
    markers[mask_mt] = '*'
    # Store in the DataFrame:
    gdf_points_units['marker'] = markers

    # In selected regions:
    col_selected = find_multiindex_column_names(
        gdf_points_units, property=['selected'])
    col_periphery_unit = find_multiindex_column_names(
        gdf_points_units, property=['periphery_unit'])
    mask = (gdf_points_units[col_selected] == 1)
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        facecolor='WhiteSmoke',
        edgecolor='k',
        return_handle=True
        )
    # Outside selected regions:
    mask = (
        (gdf_points_units[col_selected] == 0) &
        (gdf_points_units[col_periphery_unit] == 1)
    )
    ax, handles_scatter_uns = scatter_units(
        ax,
        gdf_points_units[mask],
        facecolor='WhiteSmoke',
        edgecolor='DimGray',
        return_handle=True
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units[col_selected] == 1
    col_geometry = find_multiindex_column_names(
        gdf_points_units, property=['geometry'])
    col_short_code = find_multiindex_column_names(
        gdf_points_units, property=['short_code'])
    col_stroke_team = find_multiindex_column_names(
        gdf_points_units, property=['stroke_team'])
    col_colour_lines = find_multiindex_column_names(
        gdf_points_units, property=['colour_lines'])
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units.loc[mask, col_geometry],
        gdf_points_units.loc[mask, col_short_code],
        gdf_points_units.loc[mask, col_stroke_team],
        s=label_size_units,
        colours=gdf_points_units.loc[mask, col_colour_lines],
        # color='k',
        # bbox=dict(facecolor='WhiteSmoke', edgecolor='r'),
    )
    # Outside selected regions:
    mask = (
        (gdf_points_units[col_selected] == 0) &
        (gdf_points_units[col_periphery_unit] == 1)
    )
    ax, handles_uns, labels_uns = draw_labels_short(
        ax,
        gdf_points_units.loc[mask, col_geometry],
        gdf_points_units.loc[mask, col_short_code],
        gdf_points_units.loc[mask, col_stroke_team],
        s=label_size_units,
        color='DimGray',
        # bbox=dict(facecolor='GhostWhite', edgecolor='DimGray'),
    )

    # Units:
    if len(labels_uns) > 0:
        section_labels = ['Selected units' + ' ' * 70 + '.',
                          'Periphery units']
        handles_lists = [
            [handles_scatter_us, handles_us],
            [handles_scatter_uns, handles_uns]
        ]
        labels_lists = [labels_us, labels_uns]
    else:
        section_labels = ['Selected units' + ' ' * 70 + '.']
        handles_lists = [
            [handles_scatter_us, handles_us]
        ]
        labels_lists = [labels_us]

    leg = create_units_legend(
        ax,
        handles_lists,
        labels_lists,
        section_labels,
        fontsize=6,
        bbox_to_anchor=[0.0, 1.0],
        loc='upper right'
        )

    ax = plot_lines_between_units(
        ax,
        gdf_lines_transfer,
        edgecolor='k'
        )

    if len(map_extent) > 0:
        # Limit to given extent:
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
    else:
        # Use default axis limits.
        pass

    ax.set_axis_off()  # Turn off axis line and numbers

    extra_artists = (leg, )  # Has to be a tuple.

    return ax, extra_artists


def plot_map_outcome(
        gdf_boundaries_lsoa,
        gdf_boundaries_regions,  # TO DO - make this optional...?
        gdf_points_units,
        ax=None,
        map_extent=[],
        boundary_kwargs={},
        draw_region_boundaries=True,
        ):
    """
    Map the selected units, containing regions, and LSOA outcomes.

    Inputs
    ------
    gdf_boundaries_lsoa       - GeoDataFrame.
    gdf_boundaries_regions    - GeoDataFrame.
    gdf_points_units          - GeoDataFrame.
    ax                        - plt.subplot().
    map_extent                - list. Axis limits [xmin, xmax, ymin, ymax].
    boundary_kwargs           - dict. Kwargs for plt.plot() for
                                catchment areas.
    draw_region_boundaries    - bool. Whether to draw background regions.

    Returns
    -------
    ax            - plt.subplot(). Input ax with bits drawn on.
    extra_artists - list. Extra legends drawn on with plt.add_artist().
    """
    label_size_units = 15

    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(10, 10))

    # LSOAs:
    # The column to use for colour selection is defined in
    # boundary_kwargs which should be set up in another function.
    ax = draw_boundaries(
        ax,
        gdf_boundaries_lsoa,
        **boundary_kwargs
        )

    # Region boundaries:
    kwargs_region_with_unit = {
        'edgecolor': 'DimGray',
        'linewidth': 0.5,
        'facecolor': 'none'
        }
    kwargs_region_with_lsoa = {
        'edgecolor': 'silver',
        'linewidth': 0.5,
        'facecolor': 'none'
        }
    kwargs_region_with_nowt = {
        'edgecolor': 'none',
        'facecolor': 'none'
        }

    if draw_region_boundaries:
        ax = draw_boundaries_by_contents(
            ax,
            gdf_boundaries_regions,
            kwargs_with_nowt=kwargs_region_with_nowt,
            kwargs_with_periphery=kwargs_region_with_lsoa,
            kwargs_with_unit=kwargs_region_with_unit,
            )

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'o')
    # Update MT units:
    col_use_mt = find_multiindex_column_names(
        gdf_points_units, property=['use_mt'])
    mask_mt = (gdf_points_units[col_use_mt] == 1)
    markers[mask_mt] = '*'
    # Store in the DataFrame:
    gdf_points_units['marker'] = markers

    # In selected regions:
    col_selected = find_multiindex_column_names(
        gdf_points_units, property=['selected'])
    mask = (gdf_points_units[col_selected] == 1)
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        facecolor='WhiteSmoke',
        edgecolor='k',
        return_handle=True
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units[col_selected] == 1
    col_geometry = find_multiindex_column_names(
        gdf_points_units, property=['geometry'])
    col_short_code = find_multiindex_column_names(
        gdf_points_units, property=['short_code'])
    col_stroke_team = find_multiindex_column_names(
        gdf_points_units, property=['stroke_team'])
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units.loc[mask, col_geometry],
        gdf_points_units.loc[mask, col_short_code],
        gdf_points_units.loc[mask, col_stroke_team],
        s=label_size_units,
        color='k',
        # bbox=dict(facecolor='WhiteSmoke', edgecolor='r'),
    )

    # Units:
    section_labels = ['Stroke units' + ' ' * 70 + '.']
    handles_lists = [[handles_scatter_us, handles_us]]
    labels_lists = [labels_us]

    leg = create_units_legend(
        ax,
        handles_lists,
        labels_lists,
        section_labels,
        fontsize=6,
        bbox_to_anchor=[0.0, 1.0],
        loc='upper right'
        )

    if len(map_extent) > 0:
        # Limit to given extent:
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
    else:
        # Use default axis limits.
        pass

    ax.set_axis_off()  # Turn off axis line and numbers

    extra_artists = (leg, )  # Has to be a tuple.

    return ax, extra_artists
