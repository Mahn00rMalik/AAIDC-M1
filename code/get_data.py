
import requests
import os 
from paths import GUTENBERG_DATA_DIR
from typing import List, Dict, Any # Import necessary types

def get_nutrition_book_data():
    url = "https://gutendex.com/books/"
    params = {"topic": "nutrition"}
    all_books_data = []

    print("Fetching nutrition book data from Gutendex API...")
    while url:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        for book in data["results"]:
            book_info = {
                "id": book.get("id"),
                "title": book.get("title"),
                "authors": [author.get("name") for author in book.get("authors", [])], # Extract author names
                "languages": book.get("languages", []),
                "bookshelves": book.get("bookshelves", []),
                "url":"https://www.gutenberg.org/cache/epub/"+ str(book.get("id"))+"/pg"+str(book.get("id"))+".txt"
            }
            all_books_data.append(book_info)
            

        url = data.get("next")  # Get the next page URL, if available
        params = {} # Clear params for subsequent paginated requests

    print(f"Found {len(all_books_data)} books in the Nutrition category.")
    return all_books_data

def download_data(all_books_data: List[Dict[str, Any]]):
      if not os.path.exists(GUTENBERG_DATA_DIR):
        os.makedirs(GUTENBERG_DATA_DIR)
        print(f"Created directory: {GUTENBERG_DATA_DIR}")

      for book_info in all_books_data:
        url = book_info.get('url')
        book_id = book_info.get('id')
        title = book_info.get('title')
        if url and book_id:  
             try:
                    
                    print(f"Downloading: {title} (ID: {book_id}) from {url}")                               
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    filename = f"pg{book_id}.txt"
                    filepath = os.path.join(GUTENBERG_DATA_DIR, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                          f.write(response.text)
                          print(f"Saved: {filepath}")
             except requests.exceptions.RequestException as e:
                   print(f"Error downloading {url}: {e}")
             except Exception as e:
                   print(f"An unexpected error occurred for {url}: {e}")
        else:
              print(f"Skipping book due to missing URL or ID: {book_info}")
                    

def main():
    nutrition_books_data = get_nutrition_book_data()
    download_data(nutrition_books_data)

if __name__ == "__main__":
    main()



