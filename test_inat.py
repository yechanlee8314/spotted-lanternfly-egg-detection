from src.data_collection.inaturalist import INaturalistClient

client = INaturalistClient()

data = client.get_observations(
    taxon_name="Lymantria dispar",
    photos="true",
    quality_grade="research",
    per_page=5,
)

print(data.keys())
print(len(data["results"]))
records = []

for obs in data["results"]:

    records.append({
        "id": obs["id"],
        "species": obs["taxon"]["name"],
        "date": obs["observed_on"],
    })

print(records)