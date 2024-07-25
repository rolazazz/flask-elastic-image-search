import time
import os
from pathlib import Path
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import urllib.request
from urllib.parse import urlparse, quote

PREFIX = "D:\\AppData\\product-images\\"
# Path where the photos will be downloaded
photos_donwload_path = Path(PREFIX)

def main():
    global args
    lst = []
    
    with open('.\image_embeddings\catalog-product-all.json',  encoding="utf8") as user_file:
        parsed_json = json.load(user_file)
    
        
    #photos = pd.read_json('.\image_embeddings\catalog-product-most-recent-25k.json', encoding="utf8")
    # Extract the IDs and the URLs of the photos
	# Get all cover images
    images = [(x['CoverImage']) for x in parsed_json]

	# Get all main gallery images
    # images = [(y) for x in parsed_json for y in x['Images'] if y['Type']==0]

    # Print some statistics
    print(f'Photos in the dataset: {len(images)}')


    # Create the thread pool
    threads_count = 32
    pool = ThreadPool(threads_count)

    opener = urllib.request.build_opener()
    opener.addheaders = [('whoiam', 'edl-worker')]
    urllib.request.install_opener(opener)
    
    # Start the download
    pool.map(download_photo, images)

    # Display some statistics
    print(f'Photos downloaded: {len(images)}')



# Function that downloads a single photo
def download_photo(image):

    # Get the URL of the photo (setting the width to 640 pixels)
    photo_url = f"https://img.edilportale.com/product-thumbs/b_{image['FileName']}"
    parsed_url = urlparse(photo_url)
    url = parsed_url.scheme + '://' + parsed_url.netloc + \
		quote(parsed_url.path)
		
    # Path where the photo will be stored
    photo_path = photos_donwload_path / os.path.basename(photo_url)

    # Only download a photo if it doesn't exist
    if not photo_path.exists() or not os.path.getsize(photo_path):
        try:
            urllib.request.urlretrieve(url, photo_path)
        except Exception as e:
            print(e)
            # Catch the exception if the download fails for some reason
            print(f"Cannot download {photo_url}")
            pass


if __name__ == '__main__':
    main()
