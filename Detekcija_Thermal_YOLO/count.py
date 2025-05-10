import os

def count_obstacles_in_folder(folder_path):
    total_obstacles = 0
    file_counts = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                lines = f.readlines()
                num_obstacles = len(lines)
                total_obstacles += num_obstacles
                file_counts[filename] = num_obstacles

    return total_obstacles, file_counts

# Example usage:
if __name__ == "__main__":
    folder = "datasets/test/labels"  # ← Change this to your folder path
    total, per_file = count_obstacles_in_folder(folder)

    print("Obstacles per file:")
    for fname, count in per_file.items():
        print(f"{fname}: {count}")
    
    print(f"Total obstacles: {total}\n")
