import pandas as pd
import requests
import os
import re
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

df = pd.read_csv("movies.csv")

POSTER_DIR = "movies\\posters"
os.makedirs(POSTER_DIR, exist_ok=True)


def make_safe_title(title):
    title = re.sub(r'[<>:"/\\|?*]', '', str(title))
    return title.replace(" ", "_")


def download_and_attach_posters(df, start=0, end=None):
    poster_paths = df.get("poster_path", [None] * len(df))

    if end is None:
        end = len(df)

    for i, row in tqdm(df.iloc[start:end].iterrows(), total=end-start):
        title = row["title"]
        movie_id = row.get("id", i)  # fallback jeśli nie ma id

        try:
            # query TMDB
            res = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": API_KEY, "query": title}
            ).json()

            if not res["results"]:
                continue

            movie = res["results"][0]

            if not movie.get("poster_path"):
                continue

            poster_path_tmdb = movie["poster_path"]
            image_url = f"https://image.tmdb.org/t/p/w500{poster_path_tmdb}"

            # filename: title_id.jpg
            safe_title = make_safe_title(title)
            filename = f"{safe_title}_ID_{movie_id}.jpg"
            file_path = os.path.join(POSTER_DIR, filename)

            # download only if not exists
            if not os.path.exists(file_path):
                img_data = requests.get(image_url).content
                with open(file_path, "wb") as f:
                    f.write(img_data)

            poster_paths[i] = file_path

        except Exception as e:
            print(f"Error for {title}: {e}")

    df["poster_path"] = poster_paths
    return df


# 🔹 zakres (możesz zmieniać)
START = 4500
END = 5000

df = download_and_attach_posters(df, START, END)

# zapis
# df.to_csv("movies_with_posters.csv", index=False)
df.to_csv("movies.csv", index=False)

print("Done ✅")