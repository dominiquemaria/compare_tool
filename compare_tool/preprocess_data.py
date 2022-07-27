import cbsodata
import ssl
import streamlit as st
import geopandas as gpd
import pyproj
import pandas as pd
import json
import numpy as np

mapboxt = 'pk.eyJ1IjoiZG9taW5pcXVlLXZkdiIsImEiOiJjbDNrYm4wbHEwNWNwM2RyczR0a2UxeXRoIn0.e-S9dzdYyw7IW6_B-WMHVA'
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

    geodata_url_ggd = geodata_url="https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=1.1.0&outputFormat=application%2Fjson&typeName=cbsgebiedsindelingen:cbs_provincie_2020_gegeneraliseerd"
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
