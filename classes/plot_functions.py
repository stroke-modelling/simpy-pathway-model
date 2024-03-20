"""
Functions for drawing maps.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ###################
# ##### HELPERS #####
# ###################
def combine_legend_sections(section_labels, handles_lists, labels_lists):
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


def set_legend_section_labels_to_bold(leg, section_labels, all_labels):
    # Set the section labels to bold (heavy) text:
    leg1_list = leg.get_texts()
    for s in section_labels:
        i = all_labels.index(s)
        leg1_list[i] = leg1_list[i].set_weight('heavy')
    return leg


def create_units_legend(
        ax,
        handles_lists,
        labels_lists,
        section_labels,
        **legend_kwargs
        ):
    # To combine the handle types, the list needs to be like:
    # [(h[0], h2[0]), (h[1], h2[1]), ...]
    # and it has to be pairs of tuple, not of list.

    handles_list = []
    for hlist in handles_lists:
        hlist = np.array(hlist).T.tolist()
        hlist = [tuple(l) for l in hlist]
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
# n.b. The following functions mostly just use plt.plot()
# but are given different wrappers anyway for the sake of
# applying some kwargs automatically.
def draw_boundaries_by_contents(
        ax,
        gdf_boundaries_regions,
        kwargs_with_nowt={},
        kwargs_with_lsoa={},
        kwargs_with_unit={}
        ):
    # Set up kwargs.
    kwargs_nowt = {
        'edgecolor': 'none',
        'facecolor': 'none',
    }
    kwargs_lsoa = {
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
    kwargs_lsoa = kwargs_lsoa | kwargs_with_lsoa
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
            **kwargs_lsoa
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
        mask_col='',
        ivt=True,
        mt=False,
        msu=False,
        return_handle=False,
        **kwargs
        ):
    """
    Draw scatter markers for IVT stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # if ivt:
    #     kwargs_dict = dict(
    #         edgecolor='none',
    #         facecolor='LightCoral',
    #         s=50,
    #         marker='o',
    #         zorder=2
    #     )
    # elif mt:
    #     kwargs_dict = dict(
    #         edgecolor='none',
    #         facecolor='LightCoral',
    #         s=300,
    #         marker='*',
    #         zorder=2
    #     )
    # elif msu:
    #     kwargs_dict = dict(
    #         edgecolor='none',
    #         facecolor='LightCoral',
    #         s=50,
    #         marker='s',
    #         zorder=2
    #     )
    # else:
    #     kwargs_dict = {}

    kwargs_dict = dict(
        edgecolor='none',
        facecolor='LightCoral',
        s=50,
        marker='o',
        zorder=2
    )

    # Overwrite default dict with user inputs:
    for key, val in kwargs.items():
        kwargs_dict[key] = val
    default_marker_size = kwargs_dict['s']

    # Only plot these units:
    if len(mask_col) > 0:
        mask = gdf[mask_col] == 1
        masked_gdf = gdf[mask]
    else:
        masked_gdf = gdf

    if return_handle:
        # Draw each point separately for use with the legend later.
        # n.b. this might not be necessary if I can instead work out
        # how to get separate objects out of a PathCollection object
        # containing multiple scatter points. TO DO - check this! ----------------

        # Need a separate call for each when there is an array of marker shapes.
        handles = []
        for row in masked_gdf.index:
            gdf_m = masked_gdf.loc[row]
            try:
                marker = gdf_m['marker']
                kwargs_dict['marker'] = marker
                if marker == '*':
                    # Make the star bigger.
                    kwargs_dict['s'] = 200
                else:
                    kwargs_dict['s'] = default_marker_size

            except KeyError:
                pass
            handle = ax.scatter(
                gdf_m.geometry.x,
                gdf_m.geometry.y,
                **kwargs_dict
                )
            handles.append(handle)
        return ax, handles
    else:
        # Draw all points in one call.
        ax.scatter(
            masked_gdf.geometry.x,
            masked_gdf.geometry.y,
            **kwargs_dict
            )
        return ax


def plot_lines_between_units(ax, gdf, **line_kwargs):
    """
    Draw lines from stroke units to their MT transfer units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores LineString objects that connect
            each stroke unit to its MT transfer unit.

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
            print(gdf_m['colour_lines'].values[0])
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


def draw_labels_short(ax, points, map_labels, leg_labels, **kwargs):
    """
    Draw labels from the geodataframe.
    """
    marker_kwargs = dict(
        # ha='center',
        # va='center',
        # bbox=dict(facecolor='w', edgecolor='k'),
        # fontsize=8,
        # weight=None,
        s=8,
        edgecolor="none",
        color='k'
    )
    # Update this with anything from the input dict:
    marker_kwargs = marker_kwargs | kwargs

    ref_s = marker_kwargs['s']

    from PIL import ImageFont
    from matplotlib import font_manager
    font = font_manager.FontProperties()#family='sans-serif')#, weight='bold')
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, marker_kwargs['s'])

    markers_for_legend = []
    labels_for_legend = []

    # Define "z" to shorten following "for" line:
    z = zip(
        points.x,
        points.y,
        map_labels,
        leg_labels
        )
    for x, y, map_label, leg_label in z:
        # Place the label slightly offset from the
        # exact hospital coordinates (x, y).
        if len(map_label) == 1:
            # Add an empty space after it.
            map_label = f'{map_label}~'

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

        m = ax.scatter(
            x, y,
            marker=r'$\mathdefault{' + f'{map_label}' + '}$',
            # marker=MarkerStyle(label, 'left', t),
            # label=leg_label,
            zorder=5,
            **marker_kwargs
        )
        markers_for_legend.append(m)
        labels_for_legend.append(leg_label)
        # ax.annotate(
        #     label,
        #     xy=(x, y),
        #     # xytext=(8, 8),
        #     textcoords='data',
        #     **label_kwargs
        #     )
    return ax, markers_for_legend, labels_for_legend


def plot_dummy_axis(fig, ax, leg, side='left'):
    """
    Add dummy invisible axis to the side of an existing axis.

    So that extra_artists are not cut off when plt.show() crops.

    For now this assumes that the new axis should be anchored to
    "upper right" for new axis on left-hand-side and
    "upper left" for new axis on right-hand-side.
    For other options, make new if/else settings below.
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

    TO DO - write me.
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
        kwargs_with_lsoa=kwargs_with_nowt,
        kwargs_with_unit=kwargs_with_unit,
        # kwargs_selected=kwargs_selected,
        # kwargs_not_selected=kwargs_not_selected
        )

    # In selected regions:
    mask = gdf_points_units['selected'] == 1
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        return_handle=True
        )
    # Outside selected regions:
    mask = ~mask
    ax, handles_scatter_uns = scatter_units(
        ax,
        gdf_points_units[mask],
        facecolor='Pink',
        return_handle=True
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units['selected'] == 1
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units[mask].geometry,
        gdf_points_units[mask].short_code,
        gdf_points_units[mask].stroke_team,
        s=label_size_units,
        color='k',
        # bbox=dict(facecolor='WhiteSmoke', edgecolor='r'),
    )
    # Outside selected regions:
    mask = ~mask
    ax, handles_uns, labels_uns = draw_labels_short(
        ax,
        gdf_points_units[mask].geometry,
        gdf_points_units[mask].short_code,
        gdf_points_units[mask].stroke_team,
        s=label_size_units,
        color='DimGray',
        # bbox=dict(facecolor='GhostWhite', edgecolor='DimGray'),
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
    section_labels = ['Regions with selected units', 'Other regions']
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
        section_labels = ['Selected units', 'Other units']
        handles_lists = [
            [handles_scatter_us, handles_us],
            [handles_scatter_uns, handles_uns]
        ]
        labels_lists = [labels_us, labels_uns]
    else:
        section_labels = ['Selected units']
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

    Properties of this map:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
    the background with each region given a different colour from
    its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
    Must contain:
    + coordinates of each feature / region boundary shape.
    + selected stroke unit file
    Output from Scenario.
    Must contain:
    + postcode
        - for unit name matching.
        - for labels on the map.
    + use_mt
        - for scatter marker choice.
    + [region]
        - region names to match the geojson file, for limiting the
        plotted areas to just those containing the stroke units.
    + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
    Output from Units.
    + from_postcode
        - for unit name matching.
    + name_nearest_mt
        - for setting up lines drawn between each stroke unit and
        its nearest MT unit.

    Result is saved as the name given in setup.file_selected_units_map.
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
        kwargs_with_lsoa=kwargs_region_with_nowt,
        kwargs_with_unit=kwargs_region_with_unit,
        # kwargs_selected=kwargs_selected,
        # kwargs_not_selected=kwargs_not_selected
        )

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'o')
    # Update MT units:
    mask_mt = (gdf_points_units['use_mt'] == 1)
    markers[mask_mt] = '*'
    # Store in the DataFrame:
    gdf_points_units['marker'] = markers

    # In selected regions:
    mask = gdf_points_units['selected'] == 1
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        # marker=gdf_points_units[mask]['marker'].to_list(),
        return_handle=True
        )
    # Outside selected regions:
    mask = gdf_points_units['selected'] == 0
    ax, handles_scatter_uns = scatter_units(
        ax,
        gdf_points_units[mask],
        # marker=gdf_points_units[mask]['marker'].to_list(),
        facecolor='Pink',
        return_handle=True
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units['selected'] == 1
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units[mask].geometry,
        gdf_points_units[mask].short_code,
        gdf_points_units[mask].stroke_team,
        s=label_size_units,
        color='k'
    )
    # Outside selected regions:
    mask = ~mask
    ax, handles_uns, labels_uns = draw_labels_short(
        ax,
        gdf_points_units[mask].geometry,
        gdf_points_units[mask].short_code,
        gdf_points_units[mask].stroke_team,
        s=label_size_units,
        color='DimGray'
    )

    # Units:
    if len(labels_uns) > 0:
        section_labels = ['Selected units', 'Other units']
        handles_lists = [
            [handles_scatter_us, handles_us],
            [handles_scatter_uns, handles_uns]
        ]
        labels_lists = [labels_us, labels_uns]
    else:
        section_labels = ['Selected units']
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

    Creates three maps.
    + "Drip & Ship" - catchment area of each IVT unit.
    + "Mothership" - catchment area of each MT unit, no IVT units.
    + "MSU" - catchment area of each MSU unit.

    Properties of all maps:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
    the background with each region given a different colour from
    its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
    Must contain:
    + coordinates of each feature / region boundary shape.
    + selected stroke unit file
    Output from Scenario.
    Must contain:
    + postcode
        - for unit name matching.
        - for labels on the map.
    + use_mt
        - for scatter marker choice.
    + [region]
        - region names to match the geojson file, for limiting the
        plotted areas to just those containing the stroke units.
    + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
    Output from Units.
    + from_postcode
        - for unit name matching.
    + name_nearest_mt
        - for setting up lines drawn between each stroke unit and
        its nearest MT unit.
    + geojson file of LSOA boundaries.
    Must contain:
    + coordinates of each LSOA boundary.
    + selected LSOA name file.
    Must contain:
    + column LSOA11CD, one row per selected LSOA.
    + national LSOA travel data.
    Must contain:
    + column LSOA11CD for name matching.
    + postcode_nearest_ivt
    + postcode_nearest_mt
    + postcode_nearest_msu

    Result is saved as the name given in setup class:
    + file_drip_ship_map
    + file_mothership_map
    + file_msu_map
    """
    label_size_units = 15

    if ax is None:
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(12, 8))

    # LSOAs in selected units catchment:
    mask = (gdf_boundaries_catchment['selected'] == 1)
    ax = draw_boundaries(
        ax,
        gdf_boundaries_catchment[mask],
        color=gdf_boundaries_catchment[mask]['colour'],
        **boundary_kwargs
        )

    # LSOAs in periphery units catchment:
    mask = (
        (gdf_boundaries_catchment['selected'] == 0) &
        (gdf_boundaries_catchment['periphery_unit'] == 1)
    )
    ax = draw_boundaries(
        ax,
        gdf_boundaries_catchment[mask],
        color=gdf_boundaries_catchment[mask]['colour_periphery'],
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
        kwargs_with_lsoa=kwargs_region_with_lsoa,
        kwargs_with_unit=kwargs_region_with_unit,
        )

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'o')
    # Update MT units:
    mask_mt = (gdf_points_units['use_mt'] == 1)
    markers[mask_mt] = '*'
    # Store in the DataFrame:
    gdf_points_units['marker'] = markers

    # In selected regions:
    mask = (gdf_points_units['selected'] == 1)
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        # marker=gdf_points_units[mask]['marker'].to_list(),
        facecolor='k',
        return_handle=True
        )
    # Outside selected regions:
    mask = (
        (gdf_points_units['selected'] == 0) &
        (gdf_points_units['periphery_unit'] == 1)
    )
    ax, handles_scatter_uns = scatter_units(
        ax,
        gdf_points_units[mask],
        # marker=gdf_points_units[mask]['marker'].to_list(),
        facecolor='DimGray',
        return_handle=True
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units['selected'] == 1
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units[mask].geometry,
        gdf_points_units[mask].short_code,
        gdf_points_units[mask].stroke_team,
        s=label_size_units,
        color='Gainsboro'
    )
    # Outside selected regions:
    mask = (
        (gdf_points_units['selected'] == 0) &
        (gdf_points_units['periphery_unit'] == 1)
    )
    ax, handles_uns, labels_uns = draw_labels_short(
        ax,
        gdf_points_units[mask].geometry,
        gdf_points_units[mask].short_code,
        gdf_points_units[mask].stroke_team,
        s=label_size_units,
        color='WhiteSmoke'
    )

    # Units:
    if len(labels_uns) > 0:
        section_labels = ['Selected units', 'Periphery units']
        handles_lists = [
            [handles_scatter_us, handles_us],
            [handles_scatter_uns, handles_uns]
        ]
        labels_lists = [labels_us, labels_uns]
    else:
        section_labels = ['Selected units']
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
        boundary_kwargs={},
        draw_region_boundaries=True,
        map_extent=[]
        ):
    """
    Map the selected units, containing regions, and catchment areas.

    UPDATE ME

    Creates three maps.
    + "Drip & Ship" - catchment area of each IVT unit.
    + "Mothership" - catchment area of each MT unit, no IVT units.
    + "MSU" - catchment area of each MSU unit.

    Properties of all maps:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
    the background with each region given a different colour from
    its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
    Must contain:
    + coordinates of each feature / region boundary shape.
    + selected stroke unit file
    Output from Scenario.
    Must contain:
    + postcode
        - for unit name matching.
        - for labels on the map.
    + use_mt
        - for scatter marker choice.
    + [region]
        - region names to match the geojson file, for limiting the
        plotted areas to just those containing the stroke units.
    + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
    Output from Units.
    + from_postcode
        - for unit name matching.
    + name_nearest_mt
        - for setting up lines drawn between each stroke unit and
        its nearest MT unit.
    + geojson file of LSOA boundaries.
    Must contain:
    + coordinates of each LSOA boundary.
    + selected LSOA name file.
    Must contain:
    + column LSOA11CD, one row per selected LSOA.
    + national LSOA travel data.
    Must contain:
    + column LSOA11CD for name matching.
    + postcode_nearest_ivt
    + postcode_nearest_mt
    + postcode_nearest_msu

    Result is saved as the name given in setup class:
    + file_drip_ship_map
    + file_mothership_map
    + file_msu_map
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
            kwargs_with_lsoa=kwargs_region_with_lsoa,
            kwargs_with_unit=kwargs_region_with_unit,
            )

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'o')
    # Update MT units:
    mask_mt = (gdf_points_units['use_mt'] == 1)
    markers[mask_mt] = '*'
    # Store in the DataFrame:
    gdf_points_units['marker'] = markers

    # In selected regions:
    mask = gdf_points_units['selected'] == 1
    ax, handles_scatter_us = scatter_units(
        ax,
        gdf_points_units[mask],
        # marker=gdf_points_units[mask]['marker'].to_list(),
        facecolor='Gainsboro',
        return_handle=True
        )

    # Label units:
    # In selected regions:
    mask = gdf_points_units['selected'] == 1
    ax, handles_us, labels_us = draw_labels_short(
        ax,
        gdf_points_units[mask].geometry,
        gdf_points_units[mask].short_code,
        gdf_points_units[mask].stroke_team,
        s=label_size_units,
        color='k'
    )

    # Units:
    section_labels = ['Stroke units']
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
