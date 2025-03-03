import subprocess
import sys
import time
import os

def run_pip_install(package):
    """Run pip install for a single package with error handling"""
    print(f"\n{'='*80}\nInstalling {package}...\n{'='*80}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Successfully installed {package}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}")
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    # Read requirements from file
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    # Parse requirements, skipping comments and empty lines
    packages = []
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            packages.append(line)
    
    print(f"Found {len(packages)} packages to install")
    
    # Install packages one by one
    successful = []
    failed = []
    
    for package in packages:
        if run_pip_install(package):
            successful.append(package)
        else:
            failed.append(package)
        # Small delay to avoid overwhelming the console
        time.sleep(0.5)
    
    # Summary
    print("\n\n" + "="*80)
    print(f"Installation Summary:")
    print(f"Successfully installed: {len(successful)}/{len(packages)}")
    if failed:
        print(f"\nFailed packages ({len(failed)}):")
        for pkg in failed:
            print(f"  - {pkg}")
    
    print("\nYou can try installing failed packages manually with:")
    print("pip install <package-name> --verbose")
    
    # Create a file with failed packages for easy retry
    if failed:
        with open("failed_packages.txt", "w") as f:
            for pkg in failed:
                f.write(f"{pkg}\n")
        print("\nFailed packages have been written to 'failed_packages.txt'")

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
