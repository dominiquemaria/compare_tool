import streamlit as st
import json
import pandas as pd

from compare_tool import preprocess_data, bivariate_map


@st.cache
def initialize():
    ## hoeft maar 1x vooraf ,  kan ook in preprocess-Data
    dataset_gemeente, dataset_provincie = preprocess_data.get_data()

    geometries_gemeentes, geometries_provincie = preprocess_data.get_cbs_polygons()
    df_gemeente = preprocess_data.merge_data_with_polygons(geometries=geometries_gemeentes, dataframe=dataset_gemeente, merge_col=['REGIO_NAAM', 'REGIO_CODE'])
    df_gemeente.to_csv('data/datafile_gemeente.csv', index=False)
    df_provincie = preprocess_data.merge_data_with_polygons(geometries=geometries_provincie, dataframe=dataset_provincie, merge_col=['REGIO_NAAM', 'REGIO_CODE'])
    df_provincie.to_csv('data/datafile_provincie.csv', index=False)

    j_file_gemeente = preprocess_data.create_geojson(data_and_geometries=df_gemeente, filepath=r"data/geojson_gemeente.json")
    j_file_provincie = preprocess_data.create_geojson(data_and_geometries=df_provincie, filepath=r"data/geojson_provincie.json")


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

    data, layout = bivariate_map.create_dataset(geoJSON=j_file, variable_x=variable_x, variable_y=variable_y, inverse_option_x=inverse_option_x, inverse_option_y=inverse_option_y)
    fig = bivariate_map.plot_figure(data, layout)

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




