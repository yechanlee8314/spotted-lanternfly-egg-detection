# downloads the images directly from the df metadata
from pathlib import Path
import requests
import os
import pandas as pd
import csv

class ImageDownloader:

    def get_response(self, source, params):
        response = requests.get(
            source, 
            params = params,
            timeout = 5)
        response.raise_for_status()
        return response
    
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
            "image_url": image_url,
        }

    print("Filtered CSV file created successfully!")

    def fetch_dataset(self, url, target_taxon, project_id, max_observations, quality_grade):
        print(f"Fetching observations for {target_taxon}...")
        payload = {
            "taxon_name": target_taxon,
            "project_id": project_id,
            "per_page": max_observations,
            "page": 1,
            "quality_grade":quality_grade,
        }
        response = self.get_response(source = url, params = payload)
            #TODO filter for error/empty response 

        # code to obtain keys of the returned dictionary
        # data = response.json()
        # data = data.get("results")
        # keys= data[0].keys()
        # print(list(keys))

    
        observations = response.json().get("results")

        if observations:
            records = []
            for obs in observations:
                records.append(self.parse_observation(obs))

            target_folder = "data/metadata" 
            os.makedirs(target_folder, exist_ok=True)
            filename = os.path.join(target_folder, "observations.csv")
            fieldnames = ["observation_id","species_name", "common_name", "image_url"]

            with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, extrasaction="ignore"
                )

                writer.writeheader()
                writer.writerows(records)

            print(f"Saved filtered data to {filename}")

            for record in records:
                self.download_image(
                    common_name=record.get("common_name"), 
                    image_url=record.get("image_url"), 
                    observation_id=record.get("observation_id")
                )


        

    def download_image(self, common_name, image_url, observation_id):
        if not image_url:
            return None

        image_folder = Path("data/raw/images/"+ common_name)
        image_folder.mkdir(parents=True, exist_ok=True)

        image_path = image_folder / f"{observation_id}.jpg"

        response = requests.get(image_url, stream=True)
        # TODO: check that the response succeeded
        
        chunk_size = 2000
        with open(image_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size):
                fd.write(chunk)
        return str(image_path)

        


if __name__ == "__main__":
    client = ImageDownloader()

    client.fetch_dataset(
        url = "https://api.inaturalist.org/v1/observations",
        target_taxon="Lymantria dispar",
        project_id="eggs-egg-coverings-of-terrestrial-invertebrates",
        max_observations = 50,
        quality_grade="research",
    )
