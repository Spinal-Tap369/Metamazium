import os
import json

def layout_to_str(cell_walls):
    """
    Convert the cell_walls (a list of lists) into a single string,
    where each row is concatenated and rows are separated by a newline.
    This string will serve as a unique signature for the maze layout.
    """
    return "\n".join("".join(str(val) for val in row) for row in cell_walls)

def filter_unique_mazes(input_path, output_path):
    # Load the maze tasks from the input JSON file.
    with open(input_path, "r") as f:
        tasks = json.load(f)
    
    unique_dict = {}
    for task in tasks:
        # Ensure cell_walls exists in the task.
        cell_walls = task.get("cell_walls")
        if cell_walls is None:
            continue
        # Create a string signature for the layout.
        signature = layout_to_str(cell_walls)
        # Only keep the first occurrence of a unique layout.
        if signature not in unique_dict:
            unique_dict[signature] = task

    unique_tasks = list(unique_dict.values())
    print(f"Found {len(unique_tasks)} unique maze layouts out of {len(tasks)} tasks.")
    
    # Save the unique tasks to the output JSON file.
    with open(output_path, "w") as f:
        json.dump(unique_tasks, f, indent=4)
    print(f"Unique maze layouts saved to {output_path}")

def main():
    # Set input and output file paths (adjust these paths as needed)
    input_path = "mazes_data_try/train_tasks.json"
    output_path = "mazes_data_try/unique_mazes.json"
    
    if not os.path.exists(input_path):
        print(f"Input file '{input_path}' does not exist.")
        return
    
    filter_unique_mazes(input_path, output_path)

if __name__ == "__main__":
    main()
