# https://chart-studio.plotly.com/~empet/15191/texas-bivariate-choropleth-assoc/#/

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go


def set_interval_value(x, a, b, inverse=False):
    # function that associate to a float x, a value encoding its position with respect to the interval [a, b]
    #  the associated values are 0, 1, 2 assigned as follows:
    if inverse:
        if x > b:
            return 0
        elif a < x <= b:
            return 1
        else:
            return 2
    else:
        if x <= a:
            return 0
        elif a < x <= b:
            return 1
        else:
            return 2


def data2color(x, y, a, b, c, d, biv_colors, inverse_option_x, inverse_option_y):
    # This function works only with a list of 9 bivariate colors, because of the definition of set_interval_value()
    # x, y: lists or 1d arrays, containing values of the two variables
    #  each x[k], y[k] is mapped to an int  value xv, respectively yv, representing its category,
    # from which we get their corresponding color  in the list of bivariate colors
    if len(x) != len(y):
        raise ValueError('the list of x and y-coordinates must have the same length')
    n_colors = len(biv_colors)
    if n_colors != 9:
        raise ValueError('the list of bivariate colors must have the length eaqual to 9')
    n = 3
    xcol = [set_interval_value(v, a, b, inverse=inverse_option_x) for v in x]
    ycol = [set_interval_value(v, c, d, inverse=inverse_option_y) for v in y]
    idxcol = [int(xc + n * yc) for xc, yc in
              zip(xcol, ycol)]  # index of the corresponding color in the list of bivariate colors
    colors = np.array(biv_colors)[idxcol]
    return list(colors)


def colorsquare(text_x, text_y, colorscale, n=3, xaxis='x2', yaxis='y2'):
    # text_x : list of n strings, representing intervals of values for the first variable or its n percentiles
    # text_y : list of n strings, representing intervals of values for the second variable or its n percentiles
    # colorscale: Plotly bivariate colorscale
    # returns the colorsquare as alegend for the bivariate choropleth, heatmap and more

    z = [[j + n * i for j in range(n)] for i in range(n)]
    n = len(text_x)
    if len(text_x) != n or len(text_y) != n or len(colorscale) != 2 * n ** 2:
        raise ValueError('Your lists of strings  must have the length {n} and the colorscale, {n**2}')

    text = [[text_x[j] + '<br>' + text_y[i] for j in range(len(text_x))] for i in range(len(text_y))]
    return go.Heatmap(x=list(range(n)),
                      y=list(range(n)),
                      z=z,
                      xaxis=xaxis,
                      yaxis=yaxis,
                      text=text,
                      hoverinfo='text',
                      colorscale=colorscale,
                      showscale=False)


def colors_to_colorscale(biv_colors):
    # biv_colors: list of n**2 color codes in hexa or RGB255
    # returns a discrete colorscale  defined by biv_colors
    n = len(biv_colors)
    biv_colorscale = []
    for k, col in enumerate(biv_colors):
        biv_colorscale.extend([[round(k / n, 2), col], [round((k + 1) / n, 2), col]])
    return biv_colorscale


@st.cache
def create_dataset(geoJSON, variable_x: list, variable_y: list, inverse_option_x, inverse_option_y):
    if len(variable_x) <= 12:
        marker_size = 10
    else:
        marker_size = 5
    jstevens = ["#e8e8e8", "#ace4e4", "#5ac8c8", "#dfb0d6", "#a5add3",
                "#5698b9", "#be64ac", "#8c62aa", "#3b4994"]

    p_thresh = np.nanpercentile(list(variable_x), [33, 66])
    h_thresh = np.nanpercentile(list(variable_y), [33, 66])

    facecolor = data2color(list(variable_x), list(variable_y),
                           a=p_thresh[0],
                           b=p_thresh[1],
                           c=h_thresh[0],
                           d=h_thresh[1],
                           biv_colors=jstevens,
                           inverse_option_x=inverse_option_x,
                           inverse_option_y=inverse_option_y
                           )

    if inverse_option_x:
        text_x = [f'{variable_x.name} > {p_thresh[1]}', f'{p_thresh[0]} <= {variable_x.name} <= {p_thresh[1]}', f'{variable_x.name} < {p_thresh[0]}']
    else:
        text_x = [f'{variable_x.name} < {p_thresh[0]}', f'{p_thresh[0]} <= {variable_x.name} <= {p_thresh[1]}',
                  f'{variable_x.name} > {p_thresh[1]}']
    if inverse_option_y:
        text_y = [f'{variable_y.name} > {h_thresh[1]}', f'{h_thresh[0]} <= {variable_y.name} <= {h_thresh[1]}', f'{variable_y.name} < {h_thresh[0]}']
    else:
        text_y = [f'{variable_y.name} < {h_thresh[0]}', f'{h_thresh[0]} <= {variable_y.name} <= {h_thresh[1]}',
                  f'{variable_y.name} > {h_thresh[1]}']

    legend = colorsquare(text_x, text_y, colors_to_colorscale(jstevens))
    legendlabels = {
        'Ervaren gezondheid (Goed / zeer goed, %)': "Afnemende ervaren gezondheid",
        'Instroomleeftijd wlz (mediaan)': "Afnemende leeftijd instroom wlz",
        'Aantal huisartsen FTEs per 1000 inwoners': "Afnemend aantal huisarts FTEs per 1000 inwoners",
        "Aantal 65+'ers": "Toenemend aantal 65+'ers"}

    legendtext_x = legendlabels[variable_x.name]
    legendtext_y = legendlabels[variable_y.name]

    n = len(jstevens)
    data = []
    fc = np.array(facecolor)

    for k in range(n):
        idx_color = np.where(fc == jstevens[k])[0]
        for i in idx_color:
            pts = []
            feature = geoJSON['features'][i]
            if feature['geometry']['type'] == 'Polygon':
                pts.extend(feature['geometry']['coordinates'][0])
                pts.append([None, None])  # mark the end of a polygon

            elif feature['geometry']['type'] == 'MultiPolygon':
                for polyg in feature['geometry']['coordinates']:
                    pts.extend(polyg[0])
                    pts.append([None, None])  # end of polygon
            else:
                raise ValueError("geometry type irrelevant for a map")

            X, Y = zip(*pts)
            data.append(dict(type='scatter',
                             x=X, y=Y,
                             fill='toself',
                             fillcolor=jstevens[k],  # facecolor[i],

                             hoverinfo='none',
                             mode='lines',
                             line=dict(width=1, color='rgb(256,256,256)'),
                             opacity=0.95)
                        )

    lons = []
    lats = []
    for k in range(len(geoJSON['features'])):
        county_coords = np.array(geoJSON['features'][k]['geometry']['coordinates'][0][0])
        m, M = county_coords[:, 0].min(), county_coords[:, 0].max()
        lons.append(0.5 * (m + M))
        m, M = county_coords[:, 1].min(), county_coords[:, 1].max()
        lats.append(0.5 * (m + M))

    range_lon = [np.array(lons).min()-0.2, np.array(lons).max()+0.1]
    range_lat = [np.array(lats).min()-0.2, np.array(lats).max()+0.1]

    all_variables = zip([geoJSON['features'][k]['id'] for k in range(len(geoJSON['features']))], list(variable_x), list(variable_y))

    hovertext = [f"Regio: {regio} <br>{variable_x.name}: {xvar} <br>{variable_y.name}: {yvar}" for (regio, xvar, yvar) in all_variables]

    county_centers = dict(type='scatter',
                          y=lats,
                          x=lons,
                          mode='markers',
                          text=hovertext,
                          marker=dict(size=marker_size, color=facecolor),
                          showlegend=False,
                          hoverinfo='text'
                          )

    data.append(county_centers)
    data.append(legend)

    legend_axis = dict(showline=False, zeroline=False, showgrid=False, ticks='', showticklabels=False)
    layout = go.Layout(#title='Texas bivariate choropleth<br> Ratio of 2016 to 2010 for population and housing',
                       # font = dict(family='Balto'),
                       showlegend=False, margin={"r": 200, "t": 0, "l": 0, "b": 0},
                       plot_bgcolor='rgba(0,0,0,0)',


                       #                                      hovermode = 'closest',
                                                            xaxis = dict(autorange=False,
                                                                         range=[3.34, 7.22], # an interval of lons that covers the Texas state
                                                                         domain=[0, 1],
                                                                         showgrid=False,
                                                                         zeroline=False,
                                                                         fixedrange=True,
                                                                         ticks='',
                                                                         showticklabels=False),
                                                            yaxis = dict(autorange=False,
                                                                         range=[50.57, 53.59], # an interval of lats that cover the Texas state
                                                                         domain=[0, 1],
                                                                         showgrid=False,
                                                                         zeroline=False,
                                                                         ticks='',
                                                                         showticklabels=False,
                                                                         fixedrange=True),

                       xaxis2=dict(legend_axis, **dict(domain=[0.1, 0.25],
                                                       anchor='y2',
                                                       side='bottom',
                                                       title=f"{legendtext_x} ->",
                                                       titlefont=dict(size=11))),
                       yaxis2=dict(legend_axis, **dict(domain=[0.735, 0.885],
                                                       title=f"{legendtext_y} ->",
                                                       anchor='x2')),
                       #              hovermode = 'closest',

                       height=1000,
                       #              mapbox=dict(center=dict(lat=52.13,lon=6.5), accesstoken=mapboxt, zoom=6, style="carto-positron")
                       )

    return data, layout


def plot_figure(data, layout):

    fig = go.Figure(data=data, layout=layout)
    # fig.show()
    return fig




