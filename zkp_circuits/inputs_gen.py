import json

def main():
    # Create the fields as specified, converting each number to string
    x = [str(0) for _ in range(10)]
    
    # Create a 4x10 array where each row is a vector of identical string numbers
    points = [
        [str(1) for _ in range(10)],
        [str(2) for _ in range(10)],
        [str(3) for _ in range(10)],
        [str(4) for _ in range(10)],
        [str(5) for _ in range(10)]
    ]
    
    data = {
        "x": x,
        "points": points,
        "d": str(1),
        "idx": str(0)
    }
    
    # Write the data to input.json file
    with open("input.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()