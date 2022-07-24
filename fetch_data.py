from pyodide.http import pyfetch, FetchResponse

async def fetch_data() -> FetchResponse:
    resp = await pyfetch("https://raw.githubusercontent.com/joaogui1/viz_veil/main/convs_normalization.pickle")

    bytes = await resp.bytes()

    with open('file.pickle', 'wb') as f:
        f.write(bytes)
    return