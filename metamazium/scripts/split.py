import os
import json

def split_unique_tasks(input_path, train_path, test_path, train_ratio=3/5):
    # Load the unique tasks from the input JSON file.
    with open(input_path, "r") as f:
        tasks = json.load(f)
    
    total_tasks = len(tasks)
    train_count = int(round(train_ratio * total_tasks))
    
    # Split tasks: first part for training, remainder for testing.
    train_tasks = tasks[:train_count]
    test_tasks = tasks[train_count:]
    
    # Save training tasks.
    with open(train_path, "w") as f:
        json.dump(train_tasks, f, indent=2)
    # Save testing tasks.
    with open(test_path, "w") as f:
        json.dump(test_tasks, f, indent=2)
    
    print(f"Total unique tasks: {total_tasks}")
    print(f"Saved {len(train_tasks)} training tasks to {train_path}")
    print(f"Saved {len(test_tasks)} testing tasks to {test_path}")

def main():
    # Define the paths (adjust as needed)
    input_path = os.path.join("mazes_data_try", "unique_mazes.json")
    train_path = os.path.join("mazes_data_try", "train_tasks.json")
    test_path = os.path.join("mazes_data_try", "test_tasks_small.json")
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    split_unique_tasks(input_path, train_path, test_path, train_ratio=3/5)

if __name__ == "__main__":
    main()
