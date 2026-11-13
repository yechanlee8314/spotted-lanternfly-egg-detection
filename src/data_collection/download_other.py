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

    def fetch_dataset(
        self, url, target_id, 
        max_observations, quality_grade, folder_name):
        print(f"Fetching observations for {target_id}...")
        payload = {
            "taxon_id": target_id,
            "per_page": max_observations,
            "page": 1,
            "quality_grade":quality_grade,
            "place_id": 51 #new jersey
        }
        response = self.get_response(source = url, params = payload)
        
            #TODO filter for error/empty response 

        # code to obtain keys of the returned dictionary
        # data = response.json()
        # data = data.get("results")
        # keys= data[0].keys()
        # print(data)

    
        observations = response.json().get("results")

        if observations:
            records = []
            for obs in observations:
                records.append(self.parse_observation(obs))

            target_folder = "data/metadata" 
            os.makedirs(target_folder, exist_ok=True)
            filename = os.path.join(target_folder, "observations.csv")
            fieldnames = ["observation_id","species_name", "common_name", "image_url", "local_image_path"]


            for record in records:
                local_image_path = self.download_image(
                    folder_name=folder_name,
                    image_url=record.get("image_url"),
                    observation_id=record.get("observation_id")
                )
                record["local_image_path"] = local_image_path
            with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
                
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, extrasaction="ignore"
                )

                writer.writeheader()
                writer.writerows(records)
            with open(filename, "r", encoding="utf-8") as csvfile:
                print(csvfile.read())
            print(f"Saved filtered data to {filename}")

        

    
    def download_image(self, folder_name, image_url, observation_id):
        if not image_url:
            return None

        image_folder = Path("data/raw/images") / folder_name
        image_folder.mkdir(parents=True, exist_ok=True)

        image_path = image_folder / f"{observation_id}.jpg"

        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Image download failed: {e}")
            return None

        with open(image_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=2000):
                if chunk:
                    fd.write(chunk)

        return str(image_path)


        

if __name__ == "__main__":
    client = ImageDownloader()

    client.fetch_dataset(
        url="https://api.inaturalist.org/v1/observations",
        target_id=47170,  #id for fungus
        # target_id=54743,
        max_observations=100,
        quality_grade="research",
        folder_name="fungi",
        #folder_name="lichen",

    )