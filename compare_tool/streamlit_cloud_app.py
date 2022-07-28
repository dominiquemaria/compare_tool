# all code from preprocess_data.py, bivariate_map.py and webapp.py into one script because streamlit cloud deployment
# because of error with importing custom modules

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import cbsodata

import ssl
import geopandas as gpd
import pyproj
import json

ssl._create_default_https_context = ssl._create_unverified_context


def get_cbs_dummydata():
    # Zoek op welke data beschikbaar is
    metadata = pd.DataFrame(cbsodata.get_meta('83765NED', 'DataProperties'))

    # Download data en verwijder spaties uit regiocodes
    data = pd.DataFrame(cbsodata.get_data('83765NED', select = ['WijkenEnBuurten','Gemeentenaam_1',"Omgevingsadressendichtheid_105", 'Codering_3', 'AantalInwoners_5']))
    data['Codering_3'] = data['Codering_3'].str.strip()
    data = data.rename(columns={'Codering_3': 'REGIO_CODE'})

    return data


def get_data():
    aantal_inwoners_per_provincie = pd.read_csv("data/indicatoren/aantalinw_totaal_65+_2020_provincie.csv", sep=';').rename(
        columns={'Mannen en vrouwen (aantal)': 'aantal_inwoners', "Regio's": "REGIO_NAAM"}).assign(
        REGIO_NAAM=lambda d: [el.replace(" (PV)", "") for el in d.REGIO_NAAM]).drop(
        columns=['Perioden', 'Geboorteland'])
    totaal = aantal_inwoners_per_provincie.loc[lambda d: d.Leeftijd == 'Totaal'].rename(columns={'aantal_inwoners': 'aantal_inwoners_alle_leeftijd'}).drop(columns=["Leeftijd"])
    ouder = aantal_inwoners_per_provincie.loc[lambda d: d.Leeftijd == '65 jaar of ouder'].rename(columns={'aantal_inwoners': 'aantal_inwoners_65_plus'}).drop(columns=["Leeftijd"])
    aantal_inwoners_per_provincie = totaal.merge(ouder, on=['REGIO_NAAM'])

    arbeidsvolume_huisartsen_per_provincie = pd.read_csv("data/indicatoren/arbeidsvolume_huisartsen_gzc_provincie_2020.csv", sep=';').drop(
        columns=['Perioden', 'AZW branches']).rename(
        columns={'Arbeidsvolume  (x 1 000)': 'aantal_fte', "Regio's": "REGIO_NAAM"}).assign(REGIO_NAAM = lambda d: [el.replace(" (PV)", "") for el in d.REGIO_NAAM], aantal_fte = lambda d: [float(el.replace(',','.')) *1000 for el in d.aantal_fte])

    df_provincie = arbeidsvolume_huisartsen_per_provincie.merge(aantal_inwoners_per_provincie, on=['REGIO_NAAM'])
    df_provincie = df_provincie.assign(aantal_fte_per_1000 = lambda d: np.round(d.aantal_fte / d.aantal_inwoners_alle_leeftijd * 1000, 3))

    ervarengezondheid = (pd.read_csv("data/indicatoren/ervarengezondheid_65plus_2020_gemeente.csv", sep=';')
                         .drop(columns=['ID', 'Persoonskenmerken', 'Marges']).rename(columns={'RegioS': 'REGIO_CODE'})
                         .assign(ErvarenGezondheidGoedZeerGoed_6=lambda d: [el.strip() for el in d.ErvarenGezondheidGoedZeerGoed_6])
                         .assign(ErvarenGezondheidGoedZeerGoed_6 = lambda d: np.where(d.ErvarenGezondheidGoedZeerGoed_6 == '.', np.nan, d.ErvarenGezondheidGoedZeerGoed_6))
                         .assign(ErvarenGezondheidGoedZeerGoed_6 = lambda d: [float(el) for el in d.ErvarenGezondheidGoedZeerGoed_6])
                         )
    instroomleeftijd_wlz = pd.read_csv("data/indicatoren/instroomleeftijdwlz_gemeente_2020.csv", sep=';').drop(columns=['Geslacht']).rename(columns={"Regio's":"REGIO_NAAM","Wmo-cliënten; gebruik Wlz aantal (aantal)":"aantal_wlz_gebruikers" })

    totaal_aantal_wlz_gebruikers = instroomleeftijd_wlz.groupby('REGIO_NAAM').sum().assign(mediaan=lambda d: d.aantal_wlz_gebruikers/2)
    cumulative = instroomleeftijd_wlz.sort_values(["REGIO_NAAM", "Persoonskenmerken"]).set_index(
        ['REGIO_NAAM', 'Persoonskenmerken']).groupby('REGIO_NAAM').cumsum().reset_index()

    df1 = cumulative.loc[lambda d: d['Persoonskenmerken'] == 'Leeftijd: 45 tot 60 jaar'].drop(
        columns=['Persoonskenmerken']).rename(columns={'aantal_wlz_gebruikers': 'aantal_wlz_gebruikers_45_60'})
    df2 = cumulative.loc[lambda d: d['Persoonskenmerken'] == 'Leeftijd: 60 tot 75 jaar'].drop(
        columns=['Persoonskenmerken']).rename(columns={'aantal_wlz_gebruikers': 'aantal_wlz_gebruikers_60_75'})
    df3 = cumulative.loc[lambda d: d['Persoonskenmerken'] == 'Leeftijd: 75 tot 85 jaar'].drop(
        columns=['Persoonskenmerken']).rename(columns={'aantal_wlz_gebruikers': 'aantal_wlz_gebruikers_75_85'})
    df4 = cumulative.loc[lambda d: d['Persoonskenmerken'] == 'Leeftijd: 85 jaar of ouder'].drop(
        columns=['Persoonskenmerken']).rename(columns={'aantal_wlz_gebruikers': 'aantal_wlz_gebruikers_85_plus'})
    wlz_gebruikers_alles = df1.merge(df2, on='REGIO_NAAM').merge(df3, on="REGIO_NAAM").merge(df4, on="REGIO_NAAM").merge(totaal_aantal_wlz_gebruikers.reset_index().drop(columns=['aantal_wlz_gebruikers']), on='REGIO_NAAM')
    instroomleeftijd_wlz = wlz_gebruikers_alles.assign(mediaan_instroomleeftijd_wlz = lambda d: np.where(
        d.mediaan < d.aantal_wlz_gebruikers_45_60, 45 + np.round(d.mediaan / (d.aantal_wlz_gebruikers_45_60/15)),
        np.where(
            d.mediaan < d.aantal_wlz_gebruikers_60_75, 60 + np.round((d.mediaan-d.aantal_wlz_gebruikers_45_60) / (d.aantal_wlz_gebruikers_60_75/15)),
            np.where(
                d.mediaan < d.aantal_wlz_gebruikers_75_85, 75+np.round((d.mediaan - d.aantal_wlz_gebruikers_60_75) / (d.aantal_wlz_gebruikers_75_85 / 10)),
                85+np.round((d.mediaan - d.aantal_wlz_gebruikers_75_85) / (d.aantal_wlz_gebruikers_85_plus/10))
            )
        )
    ))[['REGIO_NAAM', 'mediaan_instroomleeftijd_wlz']]

    crosswalk = pd.read_csv("data/GM_naar_GGD_en_PV.csv", sep=';')
    crosswalk_gemeente = crosswalk[["Codes en namen van gemeenten/Code (code)", "Codes en namen van gemeenten/Naam (naam)"]]
    crosswalk_gemeente.columns = ['REGIO_CODE','REGIO_NAAM']
    crosswalk_gemeente = crosswalk_gemeente.assign(REGIO_CODE= lambda d: [el.strip() for el in d.REGIO_CODE],
                              REGIO_NAAM= lambda d: [el.strip() for el in d.REGIO_NAAM])
    crosswalk_provincie = crosswalk[['Lokaliseringen van gemeenten/Provincies/Code (code)',
                                     'Lokaliseringen van gemeenten/Provincies/Naam (naam)']]
    crosswalk_provincie=crosswalk_provincie.loc[~crosswalk_provincie.duplicated()]
    crosswalk_provincie.columns = ['REGIO_CODE', 'REGIO_NAAM']
    crosswalk_provincie = crosswalk_provincie.assign(REGIO_CODE=lambda d: [el.strip() for el in d.REGIO_CODE],
                                                   REGIO_NAAM=lambda d: [el.strip() for el in d.REGIO_NAAM])

    instroomleeftijd_wlz = instroomleeftijd_wlz.merge(crosswalk_gemeente, on='REGIO_NAAM') # hier vallen er een paar weg
    ervarengezondheid = ervarengezondheid.merge(crosswalk_gemeente, on='REGIO_CODE')
    df_gemeente = ervarengezondheid.merge(instroomleeftijd_wlz, on=["REGIO_CODE", "REGIO_NAAM"])
    df_gemeente = df_gemeente.rename(columns={'ErvarenGezondheidGoedZeerGoed_6': 'Ervaren gezondheid (Goed / zeer goed, %)',
                                                "mediaan_instroomleeftijd_wlz": "Instroomleeftijd wlz (mediaan)"})

    df_provincie = df_provincie.merge(crosswalk_provincie, on="REGIO_NAAM").assign(REGIO_NAAM=lambda d: np.where(d.REGIO_NAAM == "Fryslân", "Friesland", d.REGIO_NAAM))
    df_provincie = df_provincie.rename(columns={'aantal_fte_per_1000': 'Aantal huisartsen FTEs per 1000 inwoners', "aantal_inwoners_65_plus": "Aantal 65+'ers"})

    return df_gemeente, df_provincie


def get_cbs_polygons():
    # Haal de kaart met gemeentegrenzen op van PDOK
    geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=1.1.0&outputFormat=application%2Fjson&typeName=cbsgebiedsindelingen:cbs_gemeente_2020_gegeneraliseerd'
    gemeentegrenzen = gpd.read_file(geodata_url)
    gemeentegrenzen = gemeentegrenzen.rename(columns={'statcode': 'REGIO_CODE', 'statnaam': 'REGIO_NAAM'})

    geodata_url_ggd ="https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=1.1.0&outputFormat=application%2Fjson&typeName=cbsgebiedsindelingen:cbs_provincie_2020_gegeneraliseerd"
    provinciegrenzen = gpd.read_file(geodata_url_ggd)
    provinciegrenzen = provinciegrenzen.rename(columns={'statcode': 'REGIO_CODE', 'statnaam': 'REGIO_NAAM'})
    return gemeentegrenzen, provinciegrenzen


def merge_data_with_polygons(geometries, dataframe: pd.DataFrame, merge_col = "REGIO_NAAM"):
    # Koppel CBS-data aan geodata met regiocodes
    map_df = pd.merge(geometries,dataframe, on = merge_col)

    # maak kaart
    map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

    return map_df


def create_geojson(data_and_geometries: pd.DataFrame, filepath=r'geojson.json', id="REGIO_NAAM"):

    # # write GeoJSON to filea
    data_and_geometries.to_file(filepath, driver="GeoJSON")

    with open(filepath) as geofile:
        j_file = json.load(geofile)

    # index geojson (st id value is equal to area code
    i = 1
    for feature in j_file["features"]:
        feature['id'] = j_file['features'][i - 1]['properties'][id]
        i += 1

    return j_file


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

                       height=1000
                       )

    return data, layout


def plot_figure(data, layout):

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        autosize=True,
        width=1000,
        height=1000)
    # fig.show()
    return fig




@st.cache
def initialize():
    ## hoeft maar 1x vooraf ,  kan ook in preprocess-Data
    dataset_gemeente, dataset_provincie = get_data()

    geometries_gemeentes, geometries_provincie = get_cbs_polygons()
    df_gemeente = merge_data_with_polygons(geometries=geometries_gemeentes, dataframe=dataset_gemeente, merge_col=['REGIO_NAAM', 'REGIO_CODE'])
    df_gemeente.to_csv('data/datafile_gemeente.csv', index=False)
    df_provincie = merge_data_with_polygons(geometries=geometries_provincie, dataframe=dataset_provincie, merge_col=['REGIO_NAAM', 'REGIO_CODE'])
    df_provincie.to_csv('data/datafile_provincie.csv', index=False)

    j_file_gemeente = create_geojson(data_and_geometries=df_gemeente, filepath=r"data/geojson_gemeente.json")
    j_file_provincie = create_geojson(data_and_geometries=df_provincie, filepath=r"data/geojson_provincie.json")


def read_geojson_from_disk(filepath, id="REGIO_NAAM"):
    with open(filepath) as geofile:
        j_file = json.load(geofile)

    # index geojson (st id value is equal to area code
    i = 1
    for feature in j_file["features"]:
        feature['id'] = j_file['features'][i - 1]['properties'][id]
        i += 1
    return j_file


def generate_map(aggregation, variable_x, variable_y):

    j_file = read_geojson_from_disk(f'data/geojson_{aggregation}.json')
    df = pd.read_csv(f"data/datafile_{aggregation}.csv")

    variable_x = (df[variable_x])
    variable_y = (df[variable_y])

    data, layout = create_dataset(geoJSON=j_file, variable_x=variable_x, variable_y=variable_y, inverse_option_x=inverse_option_x, inverse_option_y=inverse_option_y)
    fig = plot_figure(data, layout)

    return fig


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    initialize() # only first time running , or with development to create disk files to read

    st.sidebar.title('Kies welke indicatoren u naast elkaar wilt weergeven')

    # Remove top margin
    st.markdown(f"""<style>
    .appview-container .main .block-container{{
            padding-top: {1.5}rem;
             }}
    </style>
        """, unsafe_allow_html=True)

    aggregation = st.sidebar.radio('Op welk niveau wilt u de indicatoren vergelijken', ["provincie", "gemeente"])

    if aggregation == "gemeente":
        options_indicatoren = ['Ervaren gezondheid (Goed / zeer goed, %)',
                               'Instroomleeftijd wlz (mediaan)']
    elif aggregation == "provincie":
        options_indicatoren = ['Aantal huisartsen FTEs per 1000 inwoners', "Aantal 65+'ers"]

    indicator1 = st.sidebar.selectbox('Kies eerste indicator', options_indicatoren)
    options_indicatoren=set(options_indicatoren)-set([indicator1])
    indicator2 = st.sidebar.selectbox('Kies tweede indicator', options_indicatoren)

    toelichtingen = {
        'Ervaren gezondheid (Goed / zeer goed, %)':'Bron: RIVM statline, 2020',
        'Instroomleeftijd wlz (mediaan)':'De data voor deze indicator komt van CBS Statline (2020, met regio-indeling 2019). '
                                         'De mediaan is benaderd vanwege de gehanteerde indeling in leeftijdscategorien van 5 jaar door CBS.',
        'Aantal huisartsen FTEs per 1000 inwoners':'De data voor deze indicator komt van CBS Statline (AZW) en betreft kwartaal 1 2020.'
                                                   'Het arbeidsvolume omvat huisartsen en gezondheidscentra. De data is dus '
                                                   'geen zuivere weerspiegeling van het aantal huisarts FTEs, maar geeft'
                                                   'een indicatie op basis van de best beschikbare openbare benadering.',
        "Aantal 65+'ers":'Bron: CBS Statline, 2020'
    }

    st.sidebar.write("**Toelichting indicatoren**")
    st.sidebar.write(f'**{indicator1}**: {toelichtingen[indicator1]}')
    st.sidebar.write(f'**{indicator2}**: {toelichtingen[indicator2]}')

    body = st.text('Loading...')

    inverse_option_x = False
    inverse_option_y = False
    if indicator1 in ['Aantal huisartsen FTEs per 1000 inwoners', 'Ervaren gezondheid (Goed / zeer goed, %)', 'Instroomleeftijd wlz (mediaan)']:
        inverse_option_x = True
    if indicator2 in ['Aantal huisartsen FTEs per 1000 inwoners', 'Ervaren gezondheid (Goed / zeer goed, %)', 'Instroomleeftijd wlz (mediaan)']:
        inverse_option_y = True

    try:
        fig = generate_map(aggregation=aggregation, variable_x=indicator1, variable_y=indicator2)
        body.plotly_chart(fig, use_container_width=True)
    except:
        body.text("This is a prototype. This combination of variables is not yet implemented.")



