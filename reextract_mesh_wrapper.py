import subprocess
import sys
import os

if __name__ == "__main__":
    while True:
        try:
            path = os.path.join(os.path.dirname(__file__), "exp_runner.py")
            subprocess.run(["python", path] + sys.argv[1:], check=True)
            print("Script completed successfully.")
            break
        except subprocess.CalledProcessError as e:
            if e.returncode == 77:
                print("Restarting the script for reextract mesh...")
            else:
                raise e
        
