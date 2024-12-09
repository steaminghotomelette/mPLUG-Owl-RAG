import lancedb
from PIL import Image

# Replace this with your LanceDB path
client = lancedb.connect('path_to_your_lancedb')

# Query the image from LanceDB (replace 'your_image.jpg' with your actual query parameter)
image_data = client.query({"path": "your_image.jpg"})

if image_data:
    # Convert the image data back to a PIL image
    img = Image.fromarray(image_data["image"])
    # Show the image
    img.show()

    # Print the image dimensions
    print(f"Image dimensions: {img.size}")
else:
    print("Image not found in the database.")
