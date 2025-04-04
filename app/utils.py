import json
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup


def get_decoding_params(gn_art_id):
    response = requests.get(f"https://news.google.com/articles/{gn_art_id}")
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    div = soup.select_one("c-wiz > div")
    return {
        "signature": div.get("data-n-a-sg"),
        "timestamp": div.get("data-n-a-ts"),
        "gn_art_id": gn_art_id,
    }


def decode_urls(articles):
    articles_reqs = [
        [
            "Fbv4je",
            f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{art["gn_art_id"]}",{art["timestamp"]},"{art["signature"]}"]',
        ]
        for art in articles
    ]
    payload = f"f.req={quote(json.dumps([articles_reqs]))}"
    headers = {"content-type": "application/x-www-form-urlencoded;charset=UTF-8"}
    response = requests.post(
        url="https://news.google.com/_/DotsSplashUi/data/batchexecute",
        headers=headers,
        data=payload,
    )
    response.raise_for_status()
    return [
        json.loads(res[2])[1] for res in json.loads(response.text.split("\n\n")[1])[:-2]
    ]


# Example usage
encoded_urls = [
    "https://news.google.com/rss/articles/CBMipgFBVV95cUxPWV9fTEI4cjh1RndwanpzNVliMUh6czg2X1RjeEN0YUctUmlZb0FyeV9oT3RWM1JrMGRodGtqTk1zV3pkNEpmdGNxc2lfd0c4LVpGVENvUDFMOEJqc0FCVVExSlRrQmI3TWZ2NUc4dy1EVXF4YnBLaGZ4cTFMQXFFM2JpanhDR3hoRmthUjVjdm1najZsaFh4a3lBbDladDZtVS1FMHFn?oc=5",
    "https://news.google.com/rss/articles/CBMi3AFBVV95cUxOX01TWDZZN2J5LWlmU3hudGZaRDh6a1dxUHMtalBEY1c0TlJSNlpieWxaUkxUU19MVTN3Y1BqaUZael83d1ctNXhaQUtPM0IyMFc4R3VydEtoMmFYMWpMU1Rtc3BjYmY4d3gxZHlMZG5NX0s1RmR2ZXI5YllvdzNSd2xkOFNCUTZTaEp3b0IxZEJZdVFLUDBNMC1wNGgwMGhjRG9HRFpRZU5BMFVIYjZCOWdWcHI1YzdoVHFWYnZSOEFwQ0NubGx3Rzd0SHN6OENKMXZUcHUxazA5WTIw?hl=en-US&gl=US&ceid=US%3Aen",
]


def get_decoded_urls(urls):
    articles_params = [
        get_decoding_params(urlparse(url).path.split("/")[-1]) for url in urls
    ]
    decoded_urls = decode_urls(articles_params)
    return decoded_urls
