import os
import json
import time
import asyncio
import requests

from typing import Any, Dict
from tqdm.auto import tqdm

REGIONS = {
    "BR1": "AMERICAS",
    "EUN1": "EUROPE",
    "EUW1": "EUROPE",
    "JP1": "ASIA",
    "KR": "ASIA",
    "LA1": "AMERICAS",
    "LA2": "AMERICAS",
    "NA1": "AMERICAS",
    "OC1": "SEA",
    "RU": "EUROPE",
    "TR1": "EUROPE",
    "PH2": "ASIA",
    "SG2": "ASIA",
    "TH2": "ASIA",
    "TW2": "ASIA",
    "VN2": "ASIA",
}


class RiotClient:
    def __init__(self, region: str, key: str = None):
        if key is None:
            key = os.getenv("RIOT_KEY")
        region = region.lower()
        self.base_url = f"https://{region}.api.riotgames.com"
        self.headers = {
            "X-Riot-Token": key,
            "Origin": "https://developer.riotgames.com",
        }

    def _get(self, suffix: str, *args, **kwargs):
        while True:
            response = requests.get(
                url=self.base_url + suffix, headers=self.headers, *args, **kwargs
            )
            if response.ok:
                return response.json()
            elif "Retry-After" in dict(response.headers):
                retry_after = int(dict(response.headers)["Retry-After"])
                print(f"{response.status_code}")
                time.sleep(retry_after)

    def get_master_league(self):
        return self._get("/tft/league/v1/master")

    def get_challenger_league(self):
        return self._get("/tft/league/v1/challenger")

    def get_grandmaster_league(self):
        return self._get("/tft/league/v1/grandmaster")

    def get_summoner_by_summoner_id(self, summoner_id: str):
        return self._get(f"/tft/summoner/v1/summoners/{summoner_id}")

    def get_match_ids_by_puuid(self, puuid: str):
        return self._get(f"/tft/match/v1/matches/by-puuid/{puuid}/ids")

    def get_match_by_match_id(self, matchId: str):
        return self._get(f"/tft/match/v1/matches/{matchId}")


def get_players():
    players = []
    for region in REGIONS:
        players_to_add = []

        client = RiotClient(region)
        players_to_add += client.get_master_league()["entries"]
        players_to_add += client.get_grandmaster_league()["entries"]
        players_to_add += client.get_challenger_league()["entries"]

        for player in players_to_add:
            player["region"] = region

        players += players_to_add
        print(len(players))

    return players


def get_puuids(players):
    def get_puuid(player: Dict[str, Any]):
        client = RiotClient(player["region"])
        return client.get_summoner_by_summoner_id(player["summonerId"])

    return [get_puuid(player) for player in tqdm(players)]


def get_match_ids(players, puuids):
    async def get_match_id(player, puuid):
        region = REGIONS[player["region"]]
        client = RiotClient(region)
        await asyncio.sleep(0.05)
        match_ids = await client.get_match_ids_by_puuid(puuid["puuid"])
        return [{"match_id": match_id, "region": region} for match_id in match_ids]

    async def get_tasks():
        return await asyncio.gather(
            *[
                get_match_id(player, puuid)
                for player, puuid in tqdm(zip(players, puuids), total=len(players))
            ]
        )

    return [i for o in asyncio.run(get_tasks()) for i in o]


def get_matches(match_ids):
    results = []

    try:
        for datum in tqdm(match_ids):
            client = RiotClient(datum["region"])
            match = client.get_match_by_match_id(datum["match_id"])
            results.append(match)
    except KeyboardInterrupt:
        pass

    return results


def main():
    datum = requests.get(
        "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json"
    ).json()
    with open("src/data/en_au.json", "w") as f:
        json.dump(datum, f)

    # players = get_players()
    # with open("src/data/players1.json", "w") as f:
    #     json.dump(players, f)

    with open("src/data/players1.json", "r") as f:
        players = json.load(f)

    puuids = get_puuids(players)
    with open("src/data/puuids1.json", "w") as f:
        json.dump(puuids, f)

    # with open("src/data/puuids.json", "r") as f:
    #     puuids = json.load(f)

    match_ids = get_match_ids(players, puuids)
    with open("src/data/match_ids1.json", "w") as f:
        json.dump(match_ids, f)

    # with open("src/data/match_ids.json", "r") as f:
    #     match_ids = json.load(f)

    matches = get_matches(match_ids)
    with open("src/data/matches.json1", "w") as f:
        json.dump(matches, f)


if __name__ == "__main__":
    main()
