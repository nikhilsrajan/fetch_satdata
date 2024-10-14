# fetch_satdata
Fetching satellite data (like duh).

Requires: Python 3.10

Documentation: https://docs.google.com/document/d/1p8PMAvoggr-aegwUBtVI-NI8M40hpBdRDYl1Mh1VIs4/edit?usp=sharing


## Setup

1. Clone the repository including the submodules
    ```
    git clone https://github.com/nikhilsrajan/fetch_satdata.git
    cd fetch_satdata
    git submodule update --init --recursive
    ```

2. Create a python virtual environment and install requirements
    ```
    [python|python3|python3.10] -m venv [env-name]
    source [env-name]/bin/activate
    pip install -r requirements.txt -r cdseutils/requirements.txt -r rsutils/requirements.txt
    ```
