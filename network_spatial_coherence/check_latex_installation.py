import subprocess
import matplotlib.pyplot as plt
def check_latex_installed():
    try:
        # Try to run `latex` command, redirect output to suppress messages
        subprocess.run(["latex", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # If the command was successful, return True
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If LaTeX is not installed, a CalledProcessError or FileNotFoundError will be raised
        return False

