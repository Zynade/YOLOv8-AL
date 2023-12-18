from ultralytics import YOLO
import yaml

def main():
    active_learning_cycle(n_start=14, n_end=14, n_step=2)


def active_learning_cycle(n_start: int, n_end: int, n_step: int):
    """
    Runs the active learning cycle for the given range of n values
    """
    # Initialize the base model
    model = YOLO('yolov8x.yaml')
    
    for n in range(n_start, n_end + 2, n_step):

        # Select n percent of the dataset
        train_file = choose_n_samples(n)
        update_train_path_in_yaml(train_file)

        results = model.train(data='coco.yaml', epochs=100, pretrained=False, project="logging", name=f"{n}-perc", save=True)
        print(f"Finished training for {n} percent of the dataset")

def choose_n_samples(n: int):
    if n % 2 == 1 or not (0 <= n <= 20): 
        print(f"Invalid value of n: {n}")
        return

    output_file_name = f"train-{n}-perc.txt"
    src_file_path = f"./coco-samples/{n}.txt"
    dest_file_path = f"./datasets/coco/{output_file_name}"


    prefix = "./images/train2017/"
    suffix = ".jpg"

    write_counter = 0
    with open(src_file_path, 'r') as f:
        lines = f.readlines()
        with open(dest_file_path, 'w') as g:
            for line in lines:
                line = line.strip()
                if line == '': continue
            
                line = prefix + line + suffix
                g.write(line + '\n')
                write_counter += 1

    print(f"Successfully wrote {write_counter} lines to {dest_file_path}")
    
    return output_file_name

def update_train_path_in_yaml(train_file: str):
    with open("coco.yaml", 'r+') as f:
        doc = yaml.safe_load(f)
        doc['train'] = train_file

        # clear the file
        f.seek(0)
        f.truncate()
        
        yaml.dump(doc, f)
    print(f"Successfully updated the train path in coco.yaml to {train_file}")

if __name__ == "__main__":
    main()