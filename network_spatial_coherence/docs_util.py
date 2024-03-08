# import webbrowser
# from pathlib import Path
# import shutil
# import importlib.resources as pkg_resources
#
#
# def access_docs(deploy_path=None):
#     docs_folder_name = 'docs/build/html'  # Adjust based on your package structure
#
#     if deploy_path:
#         destination = Path(deploy_path) / 'docs'
#         # Ensure the destination directory exists
#         destination.mkdir(parents=True, exist_ok=True)
#
#         # Using importlib.resources, extract the entire html directory (Python 3.9+)
#         with pkg_resources.files('network_spatial_coherence') as pkg_path:
#             docs_src_path = pkg_path / docs_folder_name
#             if docs_src_path.exists():
#                 shutil.copytree(docs_src_path, destination, dirs_exist_ok=True)
#                 print(f"Documentation deployed to {destination}")
#             else:
#                 print(f"Documentation source not found in package at {docs_src_path}")
#     else:
#         # For viewing, find the package path to the html documentation
#         with pkg_resources.files('network_spatial_coherence') as pkg_path:
#             docs_path = pkg_path / docs_folder_name / 'index.html'
#             if docs_path.exists():
#                 webbrowser.open(docs_path.as_uri())
#                 print("Opening documentation in web browser...")
#             else:
#                 print(f"Documentation file not found at {docs_path}")

import os
from pathlib import Path
import shutil
import importlib.resources as pkg_resources
import webbrowser


def copy_docs(src, dst):
    """
    Copies all files and directories from src to dst.
    """
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_dir():
            # Recursively copy subdirectory
            copy_docs(item, dst / item.name)
        else:
            # Copy file
            shutil.copy2(item, dst / item.name)


def access_docs(deploy_path=None):
    docs_folder_name = 'docs/build/html'

    with pkg_resources.files('network_spatial_coherence') as pkg_path:
        docs_src_path = pkg_path / docs_folder_name
        print(f"Attempting to copy documentation from: {docs_src_path}")

        if deploy_path:
            destination = Path(deploy_path) / 'docs'
            print(f"Destination path: {destination}")

            # Use the custom copy function
            copy_docs(docs_src_path, destination)
            print(f"Documentation deployed to {destination}")
        else:
            docs_index_path = docs_src_path / 'index.html'
            if docs_index_path.exists():
                webbrowser.open(docs_index_path.as_uri())
                print("Opening documentation in web browser...")
            else:
                print(f"Documentation file not found at {docs_index_path}")
