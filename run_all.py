import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

scripts = [
    f for f in os.listdir(PROJECT_DIR)
    if f.endswith(".py") and f != os.path.basename(__file__)
]

scripts.sort()

print(f"Found {len(scripts)} scripts to run: {scripts}\n")

# Dictionary to store metrics
summary = {}

for script in scripts:
    print(f"\n▶️ Running {script} ...\n")
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=True,
            text=True
        )
        output = result.stdout
        print(output)

        # Parse output for metrics (simple examples)
        if "Accuracy:" in output:
            for line in output.splitlines():
                if "Accuracy:" in line:
                    accuracy = float(line.split("Accuracy:")[1].strip())
                    summary[script] = {"Accuracy": accuracy}
        elif "Silhouette Score:" in output:
            for line in output.splitlines():
                if "Silhouette Score:" in line:
                    score = float(line.split("Silhouette Score:")[1].strip())
                    summary[script] = {"Silhouette Score": score}
        elif "Explained variance ratio" in output:
            for line in output.splitlines():
                if "Explained variance ratio" in line:
                    var = float(line.split(":")[1].strip())
                    summary[script] = {"Explained Variance": var}
        elif "BIC:" in output:
            bic, sil = None, None
            for line in output.splitlines():
                if "BIC:" in line:
                    bic = float(line.split("BIC:")[1].strip())
                if "Silhouette Score:" in line:
                    sil = float(line.split("Silhouette Score:")[1].strip())
            summary[script] = {"BIC": bic, "Silhouette Score": sil}
        else:
            summary[script] = {"Status": "Completed"}

    except subprocess.CalledProcessError as e:
        print(f"❌ Script {script} failed:\n{e.stderr}")
        summary[script] = {"Status": "Failed"}

# Print summary table
print("\n\n===== SUMMARY OF ALL ALGORITHMS =====\n")
for script, metrics in summary.items():
    print(f"{script}: {metrics}")
