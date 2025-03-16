# ephemeris.py
# a simple script to generate ephemeris data for the planet sim using JPL's Horizons API

import requests
import re
import datetime


# example = r"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='499'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='OBSERVER'&CENTER='500@399'&START_TIME='2006-01-01'&STOP_TIME='2006-01-20'&STEP_SIZE='1%20d'&QUANTITIES='1,9,20,23,24,29'"

URL = r"https://ssd.jpl.nasa.gov/api/horizons.api"

def ephem_query(command: str, center: str, start_time: str, stop_time: str, step_size: str) -> str:
    # set up query parameters
    query_params = {
        "format": "text",        
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
    }
    query_params["COMMAND"] = command
    query_params["CENTER"] = center
    query_params["START_TIME"] = start_time
    query_params["STOP_TIME"] = stop_time
    query_params["STEP_SIZE"] = step_size

    url = URL + "?" + "&".join(k+"="+v for k, v in query_params.items())
    res = requests.get(url)

    assert res.status_code == 200, "Failed to get ephemeris data"
    return res.text


def ephem_query_single_day(command: str, center: str, date: str) -> str:
    stop_date = datetime.date.fromisoformat(date) + datetime.timedelta(days=1)
    return ephem_query(command, center, date, str(stop_date), '36h')


def print_major_body_codes():
    print(ephem_query("'mb'", "x", "x", "x", "x"))


# query vector regex 
vector_table_pattern = re.compile(r"\$\$SOE(.*)\$\$EOE", re.M | re.DOTALL)
vt_element = lambda el: el + r"=\s?(\S+)"

# gets the position and velocity of an ephem query 
def ephem_vectors(query: str) -> tuple[list[float], list[float]]:
    vt = re.search(vector_table_pattern, query)
    data = []
    for e in ["X ", "Y ", "Z ", "VX", "VY", "VZ"]:
        m = re.search(vt_element(e), vt.group(0))
        str_val = m.group(1)
        data.append(float(str_val))

    return (data[:3], data[3:])


# rotation_rate_pattern = re.compile(r"(Rot)", re.M | re.DOTALL)

# def ephem_rotation_rate(query: str) -> float:
#     m = re.search(rotation_rate_pattern, query)
#     # return float(m.group(0))
#     print(m.groups())


# gets the position and velocity of an astronomical body in relation to another astronomical body
def get_ephem_values(target: str, center: str, date: str) -> tuple[list[float], list[float], float]:
    query = ephem_query_single_day(target, center, date)
    pos, vel = ephem_vectors(query)
    # rot = ephem_rotation_rate(query)
    return (pos, vel)


planet_codes = {
    "Sun": "10",
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Moon": "301",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699",
    "Uranus": "799",
    "Neptune": "899",
}
date = "2025-03-01"
solar_system_barycenter = "'500@0'"

for planet, code in planet_codes.items():
    pos, vel = get_ephem_values(code, solar_system_barycenter, date)
    print(planet, end = ";;;")
    for x in pos:
        print(x, end = ";")
    for x in vel:
        print(x, end = ";;;")
    print()