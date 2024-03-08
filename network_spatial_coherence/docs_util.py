import importlib.resources as pkg_resources
import webbrowser
from shutil import copytree, rmtree
from pathlib import Path
import tempfile


def access_docs(deploy_path=None):
    # Assuming 'network_spatial_coherence' is your package name and 'docs' is directly under it
    package_name = 'network_spatial_coherence'

    # Temporarily create a path for the documentation
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Assuming the documentation HTML files are included in your package_data
        doc_files = pkg_resources.contents(package_name)
        docs_path = Path(tmpdirname) / "docs"
        docs_path.mkdir(parents=True, exist_ok=True)

        # Copy all documentation files to the temporary directory
        for file in doc_files:
            if file.startswith('docs/build/html/'):  # Adjust this path
                content = pkg_resources.read_binary(package_name, file)
                (docs_path / Path(file).name).write_bytes(content)

        if deploy_path:
            # Copy the documentation from the temporary directory to the specified path
            destination = Path(deploy_path) / 'docs'
            if destination.exists():
                rmtree(destination)  # Remove the existing destination to avoid conflicts
            copytree(docs_path, destination)
            print(f"Documentation deployed to {destination}")
        else:
            # Open the index.html in the web browser from the temporary directory
            index_path = docs_path / 'index.html'  # Adjust if your index.html is deeper in the structure
            webbrowser.open('file://' + str(index_path))
            print("Opening documentation in web browser...")
