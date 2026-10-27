
from pathlib import Path
import shutil

import pandas as pd
import requests


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


    def fetch_inaturalist_dataset(
        self,
        target_taxon,
        project_id,
        max_observations=5,
    ):
        print(f"Fetching observations for {target_taxon}...")

        params = {
            "taxon_name": target_taxon,
            "project_id": project_id,
            "per_page": max_observations,
            "page": 1,
        }

        response = self.get_observations(**params)
        print(response.get("results"))
        observations = response.get("results", [])
        
        records = [
            self.parse_observation(obs)
            for obs in observations
        ]

        df = pd.DataFrame(records)

        if df.empty:
            print("No observations returned.")
            return df

        image_dir = Path("data/raw/images")
        image_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        download_results = []

        for _, row in df.iterrows():
            observation_id = row["observation_id"]
            url = row["image_url"]

            if pd.isna(url) or not url:
                image_paths.append(None)
                download_results.append(False)
                print(f"Skipping observation {observation_id}: no image URL")
                continue

            image_path = image_dir / f"{observation_id}.jpg"

            # Reuse an existing file if it has already been downloaded.
            if image_path.exists() and image_path.stat().st_size > 0:
                success = True
            else:
                success = self.download_image_stream(
                    url, image_path
                )

            image_paths.append(
                str(image_path) if success else None
            )
            download_results.append(success)

            if not success:
                print(f"Download failed: observation {observation_id}")

        df["local_image_path"] = image_paths
        df["download_success"] = download_results

        output_filename = (
            f"inat_dataset_{target_taxon.lower().replace(' ', '_')}.csv"
        )
        self.save_df(df, output_filename)

        successful = sum(download_results)
        print(
            f"Finished: {len(df)} observations, "
            f"{successful} images available."
        )
        return df


    def parse_observation(self, obs):
        photos = obs.get("photos", [])

        image_url = None
        if photos:
            image_url = photos[0].get("url")
            if image_url:
                image_url = image_url.replace(
                    "/square.", "/large."
                )

        taxon = obs.get("taxon") or {}

        return {
            "observation_id": obs.get("id"),
            "species_name": taxon.get("name"),
            "common_name": taxon.get("preferred_common_name"),
            "observed_on": obs.get("observed_on"),
            "quality_grade": obs.get("quality_grade"),
            "image_url": image_url,
        }

    def save_df(self, df, filename):
        output_dir = Path("data/metadata")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        print(f"Saved metadata to {output_path}")

    @staticmethod
    def download_image_stream(url, dest_path):
        if not isinstance(url, str) or not url.strip():
            return False

        try:
            with requests.get(
                url, stream=True, timeout=20
            ) as response:
                response.raise_for_status()

                with open(dest_path, "wb") as file:
                    shutil.copyfileobj(
                        response.raw, file
                    )

            # Reject empty files.
            if dest_path.stat().st_size == 0:
                dest_path.unlink()
                return False

            return True

        except (requests.RequestException, OSError) as error:
            print(f"Image download error: {error}")
            if dest_path.exists():
                dest_path.unlink()
            return False


if __name__ == "__main__":
    client = INaturalistClient()

    client.fetch_inaturalist_dataset(
        target_taxon="Lymantria dispar",
        project_id="eggs-egg-coverings-of-terrestrial-invertebrates",
        max_observations=2,
    )
