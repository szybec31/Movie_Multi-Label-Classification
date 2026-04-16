import pandas as pd
import requests
import os
import re
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
df = pd.read_csv("movies.csv")

os.makedirs("movies/posters", exist_ok=True)

#Download posters for the first 20 records of the movies.csv file
for i, row in df.head(10).iterrows():
    movie_name = row["title"]
    try:
        # Search for a movie
        res = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": API_KEY, "query": movie_name}
        ).json()

        if not res["results"]:
            print(f"No results for: {movie_name}")
            continue

        movie = res["results"][0]

        if not movie.get("poster_path"):
            print(f"No poster for: {movie_name}")
            continue

        # Build image URL since API returns only image path
        poster_path = movie["poster_path"]
        image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"

        # Download and save the image with a correct name into the set directory
        img_data = requests.get(image_url).content
        movie_name = re.sub(r'[<>:"/\\|?*]', '', movie_name)
        safe_name = movie_name.replace(" ", "_")
        file_name = f"movies/posters/{safe_name}.jpg"

        with open(file_name, "wb") as f:
            f.write(img_data)

        print(f"Saved: {file_name}")

    except Exception as e:
        print(f"Error for {movie_name}: {e}")



