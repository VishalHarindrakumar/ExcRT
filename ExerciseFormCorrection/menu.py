import subprocess
import os
import shutil
import sys
import pkg_resources

REQUIRED_PYTHON_VERSION = (3, 8)
REQUIRED_SKLEARN_VERSION = "1.1.2"

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_console_width():
    return shutil.get_terminal_size().columns

def center_text(text):
    console_width = get_console_width()
    return text.center(console_width)

def check_python_version():
    return sys.version_info >= REQUIRED_PYTHON_VERSION

def check_sklearn_version():
    try:
        sklearn_version = pkg_resources.get_distribution("scikit-learn").version
        return sklearn_version == REQUIRED_SKLEARN_VERSION
    except pkg_resources.DistributionNotFound:
        return False

def show_instructions():
    clear_console()
    print(center_text("=" * 40))
    print(center_text("REQUIRED SETUP MISSING"))
    print(center_text("=" * 40))
    print(center_text(f"Python 3.8 or higher is required. You are using {sys.version.split()[0]}"))
    print(center_text(f"scikit-learn version 1.1.2 is required."))
    print(center_text("Please run the following commands:"))
    print(center_text(f"pip install python=={REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}"))
    print(center_text(f"pip install scikit-learn=={REQUIRED_SKLEARN_VERSION}"))
    print("=" * 40)

def show_menu():
    clear_console()
    print("\n" + center_text("="*30))
    print(center_text("Exercise Selection Menu"))
    print(center_text("="*30))
    print(center_text("1. Bicep Curl"))
    print(center_text("2. Plank"))
    print(center_text("3. Squat"))
    print(center_text("4. Exit"))
    print(center_text("="*30) + "\n")

def run_exercise(choice):
    if choice == '1':
        subprocess.run(["python", "bicep_audio_update.py"])
    elif choice == '2':
        subprocess.run(["python", "plank_audio_update.py"])
    elif choice == '3':
        subprocess.run(["python", "squat_audio_update.py"])
    elif choice == '5':
        subprocess.run(["python", "lunge.py"])
    elif choice == '4':
        print(center_text("\nThank you for using the exercise menu! Goodbye.\n"))
        return False
    else:
        print(center_text("\nInvalid choice. Please try again.\n"))
    return True

if __name__ == "__main__":
    if not check_python_version() or not check_sklearn_version():
        show_instructions()
    else:
        while True:
            show_menu()
            user_choice = input(center_text("Enter the number of the exercise: "))
            if not run_exercise(user_choice):
                break
