import requests
import pandas as pd
from pathlib import Path

class INaturalistClient:

    BASE_URL = "https://api.inaturalist.org/v1"

    def __init__(self):
        self.session = requests.Session()

    def get_observations(self, **params): 

        response = self.session.get(
            f"{self.BASE_URL}/observations",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()
    

    def fetch_inaturalist_dataset(self, target_taxon, target_place = None,
        max_observations=5):
        print(f"Fetching data for {target_taxon} in {target_place}...")

        # 2. Query the iNaturalist API
        # 'verifiable=true' filters for quality data (research grade or needs ID)
        response = self.get_observations(
            taxon_name=target_taxon,
            # place_id=target_place,
            per_page=max_observations,  # Maximizes the records retrieved per API call
        )
        print("Sending API request...")
        print(response.keys())
        # 3. Parse and structure the JSON response into records
        parsed_records = []
        for obs in response.get("results", []):
            # Extract location coordinates safely
            location = obs.get("location")

            lat, lon = None, None
            if location:
                lat_str, lon_str = location.split(",")
                lat = float(lat_str)
                lon = float(lon_str)
            # Build data dictionary row by row
            record = {
                "observation_id": obs.get("id"),
                "species_name": obs.get("taxon", {}).get("name"),
                "common_name": obs.get("taxon", {}).get("preferred_common_name"),
                "observed_on": obs.get("observed_on_details", {}).get("date"),
                "latitude": lat,
                "longitude": lon,
                "quality_grade": obs.get("quality_grade"),
                "user": obs.get("user", {}).get("login"),
                "image_url": (
                    obs.get("photos", [{}])[0].get("url")
                    if obs.get("photos")
                    else None
                ),
            }
            parsed_records.append(record)
        print(f"Found {len(response['results'])} observations")
        # 4. Convert the structured list to a Pandas DataFrame
        df = pd.DataFrame(parsed_records)

        # 5. Export dataset as a CSV file
        output_dir = Path("data/metadata")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = output_dir / (
            f"inat_dataset_{target_taxon.lower().replace(' ', '_')}.csv"
        )
        df.to_csv(output_filename, index=False)

        print(f"Success! Saved {len(df)} records to '{output_filename}'")
        return df


if __name__ == "__main__":

    client = INaturalistClient()

    client.fetch_inaturalist_dataset(
        target_taxon="Lymantria dispar",
        max_observations=5,
    )