#  python3.10 -m pip install streamlit geoip2 watchdog pycountry pandas
#  streamlit run country_ipfinder.py
#
# Download: https://code.vt.edu/hsinyu/edx-platform-release/-/raw/bd08bc7bdc21d09851e8c52a435a34222b4af767/common/static/data/geoip/GeoLite2-Country.mmdb
# Google search: download GeoLite2-Country filetype:mmdb

import streamlit as st
import geoip2.database
import ipaddress
import tempfile
import os
import pycountry
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Helper: Build country code-name mapping
country_dict = {country.alpha_2: country.name for country in pycountry.countries}
country_list = sorted([(code, name) for code, name in country_dict.items()], key=lambda x: x[1])

st.title("Country IP Finder with MaxMind GeoLite2")

# Upload MaxMind database
mmdb_file = st.file_uploader("Upload your MaxMind GeoLite2 Country/City .mmdb file", type=["mmdb"])

# Select country
country_selection = st.selectbox(
    "Select a country:",
    options=country_list,
    format_func=lambda x: f"{x[1]} ({x[0]})"
)
selected_country_code = country_selection[0]
selected_country_name = country_selection[1]

# Input IP ranges (CIDR format, one per line)
ip_ranges_text = st.text_area(
    "Enter IP ranges (CIDR notation, one per line):",
    value="31.9.0.0/16\n46.32.0.0/13\n77.44.0.0/15"
)

def lookup_ips(mmdb_path, ip_list, country_code):
    found = []
    with geoip2.database.Reader(mmdb_path) as reader:
        for ip in ip_list:
            try:
                response = reader.country(str(ip))
                if response.country.iso_code == country_code:
                    found.append({"ip": str(ip), "country_code": country_code})
            except geoip2.errors.AddressNotFoundError:
                continue
    return found

if st.button(f"Find IPs in {selected_country_name}") and mmdb_file and ip_ranges_text:
    ip_ranges = [line.strip() for line in ip_ranges_text.splitlines() if line.strip()]
    all_ips = []
    for cidr in ip_ranges:
        try:
            net = ipaddress.ip_network(cidr)
            all_ips.extend(list(net))
        except ValueError:
            st.warning(f"Invalid CIDR: {cidr}")
            continue

    # Save uploaded mmdb to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mmdb") as tmp:
        tmp.write(mmdb_file.read())
        mmdb_path = tmp.name

    # Split IPs for threading
    num_workers = multiprocessing.cpu_count()
    chunk_size = max(1, len(all_ips) // num_workers)
    ip_chunks = [all_ips[i:i + chunk_size] for i in range(0, len(all_ips), chunk_size)]

    found_ips = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(lookup_ips, mmdb_path, chunk, selected_country_code) for chunk in ip_chunks]
        for future in as_completed(futures):
            found_ips.extend(future.result())

    os.remove(mmdb_path)  # Clean up temp file

    st.success(f"Found {len(found_ips)} IPs in {selected_country_name}.")
    if found_ips:
        df = pd.DataFrame(found_ips)
        st.dataframe(df.head(100), use_container_width=True)

        # Download as TXT
        st.download_button(
            label=f"Download as TXT",
            data="\n".join(ip['ip'] for ip in found_ips),
            file_name=f"{selected_country_name.replace(' ', '_').lower()}_ips.txt",
            mime="text/plain"
        )
        # Download as CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            label=f"Download as CSV",
            data=csv_data,
            file_name=f"{selected_country_name.replace(' ', '_').lower()}_ips.csv",
            mime="text/csv"
        )
    else:
        st.info("No IPs found for the selected country in the provided ranges.")
else:
    st.info("Upload a MaxMind database, select a country, and enter IP ranges to begin.")

st.markdown("""
---
*Contact [Manceps](mailto:info@manceps.com)*
*Powered by [Streamlit](https://streamlit.io/), [MaxMind GeoLite2](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data?lang=en), [pycountry](https://pypi.org/project/pycountry/), and [pandas](https://pandas.pydata.org/).*
""")

