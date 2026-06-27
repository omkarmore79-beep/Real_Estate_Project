import os

import fitz

from config import IMAGE_FOLDER


def extract_images_from_pdf(
    pdf_path,
    output_folder=IMAGE_FOLDER,
    image_base_path="storage/images",
):
    """Render each brochure page as an image and return JSON-ready metadata stubs.

    Rendering full pages is more reliable for brochure search than extracting
    embedded image objects. Brochure pages are often assembled from many small
    image fragments, icons, and rotated assets, so embedded extraction can return
    cropped or incorrectly oriented pieces instead of the actual plan page.
    """
    os.makedirs(output_folder, exist_ok=True)

    extracted_images = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            image_id = f"page_{page_number}"
            filename = f"page_{page_number}.png"
            output_path = os.path.join(output_folder, filename)

            pix = page.get_pixmap(dpi=160, alpha=False)
            pix.save(output_path)

            extracted_images.append(
                {
                    "image_id": image_id,
                    "page": page_number,
                    "image_path": f"{image_base_path}/{filename}",
                    "local_path": output_path,
                }
            )

    return extracted_images
