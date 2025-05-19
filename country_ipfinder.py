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
import requests

# RIR delegation file URLs
RIR_URLS = [
    "https://ftp.ripe.net/pub/stats/ripencc/delegated-ripencc-latest",
    "https://ftp.arin.net/pub/stats/arin/delegated-arin-extended-latest",
    "https://ftp.apnic.net/pub/stats/apnic/delegated-apnic-latest",
    "https://ftp.lacnic.net/pub/stats/lacnic-latest",
    "https://ftp.afrinic.net/pub/stats/afrinic/delegated-afrinic-latest"
]

def download_delegation_files():
    files = []
    for url in RIR_URLS:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(r.content)
                tmp.close()
                files.append(tmp.name)
        except Exception as e:
            pass  # Ignore download errors for now
    return files

def parse_delegation(file_path, country_code, debug_log):
    cidrs = []
    with open(file_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('|')
            if len(parts) > 6 and parts[1].upper() == country_code and parts[2] == 'ipv4':
                debug_log.append(f"Matched line: {line.strip()}")
                start_ip = parts[3]
                count = int(parts[4])
                try:
                    # Compute last IP in the range
                    start = ipaddress.IPv4Address(start_ip)
                    end = ipaddress.IPv4Address(int(start) + count - 1)
                    for net in ipaddress.summarize_address_range(start, end):
                        cidrs.append(net.with_prefixlen)
                        debug_log.append(f"Parsed: {start} - {end} -> {net.with_prefixlen}")
                except Exception as e:
                    debug_log.append(f"Failed to parse {start_ip}/{count}: {e}")
            elif len(parts) > 6 and parts[1].upper() == country_code:
                debug_log.append(f"Non-ipv4 match: {line.strip()}")
    return cidrs

def get_country_cidrs(country_code, debug_log):
    files = download_delegation_files()
    cidrs = set()
    for file_path in files:
        cidrs.update(parse_delegation(file_path, country_code, debug_log))
        os.remove(file_path)
    return sorted(cidrs)

# Helper: Build country code-name mapping
country_dict = {country.alpha_2: country.name for country in pycountry.countries}
country_list = sorted([(code, name) for code, name in country_dict.items()], key=lambda x: x[1])
dropdown_options = [("", "Select a country...")] + country_list

st.title("Country IP Finder (RIR Auto CIDR, MaxMind)")

mmdb_file = st.file_uploader("Upload your MaxMind GeoLite2 Country/City .mmdb file", type=["mmdb"])

country_selection = st.selectbox(
    "Select a country:",
    options=dropdown_options,
    format_func=lambda x: f"{x[1]} ({x[0]})" if x[0] else x[1]
)
selected_country_code = country_selection[0]
selected_country_name = country_selection[1]

if 'cidr_cache' not in st.session_state:
    st.session_state['cidr_cache'] = {}
if 'debug_log' not in st.session_state:
    st.session_state['debug_log'] = []

debug_log = st.session_state['debug_log']

if selected_country_code and selected_country_code not in st.session_state['cidr_cache']:
    debug_log.clear()
    with st.spinner(f"Fetching IP blocks for {selected_country_name}..."):
        cidrs = get_country_cidrs(selected_country_code, debug_log)
        st.session_state['cidr_cache'][selected_country_code] = cidrs
elif selected_country_code:
    cidrs = st.session_state['cidr_cache'][selected_country_code]
else:
    cidrs = []

with st.expander("Show Delegation File Debug Log"):
    st.text("\n".join(debug_log[-1000:]))

ip_ranges_text = st.text_area(
    "IP ranges (CIDR notation, one per line):",
    value="\n".join(cidrs) if cidrs else "",
    height=200,
    disabled=not selected_country_code
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

if st.button(f"Find IPs in {selected_country_name}") and mmdb_file and ip_ranges_text and selected_country_code:
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
elif not selected_country_code:
    st.info("Please select a country to fetch its IP blocks.")

st.markdown("""
---
*Contact [Manceps](mailto:info@manceps.com)*

*Powered by [Streamlit](https://streamlit.io/), [MaxMind GeoLite2](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data?lang=en), [pycountry](https://pypi.org/project/pycountry/), and [pandas](https://pandas.pydata.org/).*
""")

