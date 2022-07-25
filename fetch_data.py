from pyodide.http import pyfetch, FetchResponse

async def fetch_data(experiment) -> FetchResponse:
    resp = await pyfetch(f"https://raw.githubusercontent.com/joaogui1/viz_veil/main/data/{experiment}.pickle")

    bytes = await resp.bytes()

    with open('file.pickle', 'wb') as f:
        f.write(bytes)
    return