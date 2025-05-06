import re
import ast
import statistics

def summarize_file(path):
    """
    Reads a file where each line is of the form:
      key: [v1, v2, v3, …]
    Handles values wrapped in np.float32(...) by removing the wrapper.
    Prints the mean and standard deviation for each key.
    """
    float32_pattern = re.compile(r'np\.float32\(\s*([^\)]+)\s*\)')

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue

            # split off the label and the list-text
            label, list_text = line.split(':', 1)
            label = label.strip()

            # remove any np.float32(...) wrappers
            cleaned = float32_pattern.sub(r'\1', list_text)

            try:
                # safely evaluate the bracketed list to a Python list of floats
                values = ast.literal_eval(cleaned.strip())
            except (SyntaxError, ValueError) as e:
                print(f"Skipping line (could not parse): {line!r}  ({e})")
                continue

            # compute statistics (needs at least two points for stdev)
            try:
                mean = statistics.mean(values)
                sd   = statistics.stdev(values) if len(values) > 1 else 0.0
            except Exception as e:
                print(f"Skipping line (could not compute statistics): {line!r}  ({e})")
                continue

            print(f"{label:10s} → mean = {mean:.6f},  sd = {sd:.6f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python summarize.py data.txt")
    else:
        summarize_file(sys.argv[1])
