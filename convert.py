import os
from multiprocessing import Pool

from pdf2image import convert_from_path

SOURCE_DIR = "./data/pdf/"
DEST_DIR = "./data/png"


source_filenames = [
    os.path.join(SOURCE_DIR, f)
    for f in os.listdir(SOURCE_DIR)
    if f.endswith(".pdf")
]


def convert_pdf(path):
    try:
        filename = os.path.basename(path)
        png_filename = filename.replace(".pdf", "")
        png_folder_path = os.path.join(DEST_DIR, png_filename)
        os.makedirs(
            png_folder_path, exist_ok=True
        )  # Create the path for the current folder
        images = convert_from_path(path, dpi=50)
        for i, image in enumerate(images):
            image.save(os.path.join(png_folder_path, f"{i}.png"))
        return (True, path)
    except Exception as e:
        print(f"Error converting {path}: {e}")
        return (False, path)


with Pool(20) as p:
    res = p.map(convert_pdf, source_filenames)
    res = [p for s, p in res if not s]
    print(f"Failed to convert {len(res)} elements : ")
    for r in res:
        print("Failed to convert : " + r)
