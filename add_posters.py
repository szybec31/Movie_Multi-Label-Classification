import os
import re

def make_safe_title(title):
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    return title.replace(" ", "_")

def attach_posters(df, poster_dir="movies\posters"):
    poster_files = set(os.listdir(poster_dir))
    status = False
    paths = []
    for title in df["title"]:
        safe = make_safe_title(title)
        filename = f"{safe}.jpg"

        if filename in poster_files:
            paths.append(os.path.join(poster_dir, filename))
            status = True

        else:
            paths.append(None)

    df["poster_path"] = paths
    return df, status