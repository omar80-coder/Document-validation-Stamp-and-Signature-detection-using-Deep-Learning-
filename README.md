import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2


plt.style.use('ggplot')

# This class keeps track of the training and validation loss values
# and helps to get the average for each epoch as well.
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model during training based on the validation mAP.
    """
    def __init__(self, best_valid_map=float('-inf')):
        """
        Initialize the SaveBestModel object with the best validation mAP observed.

        Parameters:
        best_valid_map (float): The best validation mAP observed so far.
        """
        self.best_valid_map = best_valid_map

    def __call__(self, model, current_valid_map, epoch, OUT_DIR):
        """
        If the current validation mAP is higher than the previously observed best, save the model.

        Parameters:
        model (torch.nn.Module): The model to save.
        current_valid_map (float): The current epoch's validation mAP.
        epoch (int): The current epoch number.
        OUT_DIR (str): Directory to save the best model.
        """
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"SAVING BEST MODEL AT EPOCH: {epoch+1}\n")
            path = f"{OUT_DIR}/best_model.pth"
            # Sauvegarder le dictionnaire de l'état du modèle
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, path)

            # Enregistrer l'artefact dans MLflow
            #mlflow.log_artifact(path, "best_model")
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return valid_loader

# Define the training tranforms.
import albumentations as A
def get_train_transform():
    return A.Compose([
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(p=0.4),
        A.MultiplicativeNoise(multiplier=(0.5, 1), per_channel=True, p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# Define the validation transforms.
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]],
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def save_model(epoch, model, optimizerr,scheduler, train_loss_list, validation_loss_list, map_50_list, map_list, OUT_DIR='outputs'):
    """
    Function to save the trained model till the current epoch, or whenever called.
    Additionally, save the training and validation loss and mAP history for continuity in tracking the model's performance.

    Parameters:
    epoch (int): Current epoch number.
    model (torch.nn.Module): Model to be saved.
    optimizer (torch.optim.Optimizer): Optimizer to be saved.
    scheduler (torch.optim.lr_scheduler): Scheduler to be saved.
    train_loss_list (list): List of training loss values.
    validation_loss_list (list): List of validation loss values.
    map_50_list (list): List of mAP@0.50 values.
    map_list (list): List of mean average precision values.
    OUT_DIR (str): Directory to save the model checkpoint.
    """
    path = f'{OUT_DIR}/last_model.pth'  # Define path for clarity and troubleshooting
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Ensure scheduler is passed and saved
        'train_loss_list': train_loss_list,
        'validation_loss_list': validation_loss_list,
        'map_50_list': map_50_list,
        'map_list': map_list,
    }, path)
     #mlflow.log_artifact(path, "model_checkpoints")

    print(f'Model and metrics saved for epoch {epoch}')  # Adjusted to display current epoch



def save_loss_plot(OUT_DIR, train_loss_list, validation_loss_list, x_label='Epochs', y_label='Loss', save_name='loss_plot'):
    """
    Function to save the training and validation loss graphs.

    Parameters:
    OUT_DIR (str): Directory to save the plot.
    train_loss_list (list): List of training loss values.
    validation_loss_list (list): List of validation loss values.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    save_name (str): Filename for the saved plot.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_list, label='Training Loss', color='blue')
    plt.plot(validation_loss_list, label='Validation Loss', color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{OUT_DIR}/{save_name}.png")
    plt.close()
    print('SAVING LOSS PLOTS COMPLETE...')

def save_mAP(OUT_DIR, map_05, map):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.
    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU (expected as list of tensors or floats).
    :param map: List containing mAP values at 0.5:0.95 IoU (expected as list of tensors or floats).
    """
    if len(map_05) > 0 and isinstance(map_05[0], torch.Tensor):
        map_05 = [m.cpu().item() for m in map_05]  # Appliquer .cpu() à chaque tenseur
    if len(map) > 0 and isinstance(map[0], torch.Tensor):
        map = [m.cpu().item() for m in map]
    # Création du graphique
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-',
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-',
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    plt.title('mAP over Epochs')
    plt.grid(True)
    figure.savefig(f"{OUT_DIR}/map.png")
    plt.close(figure)  # Fermez le graphique pour libérer de la mémoire


from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import json
import os
import math
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
class InvoiceDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file, image_dir, width, height, classes, transforms=None):
        self.annotations = [json.loads(line) for line in open(jsonl_file, 'r')]
        self.image_dir = image_dir
        self.height = height
        self.width = width
        self.classes = classes
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
            annotation = self.annotations[idx]
            img_path = os.path.join(self.image_dir, annotation['filename'])
            original_img = Image.open(img_path).convert("RGB")

            # Obtenir les dimensions originales de l'image
            original_width, original_height = original_img.size
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR en RGB
            image = image.astype(np.float32) / 255.0  # Convertir en float32 et normaliser
            image_resized = cv2.resize(image, (self.width, self.height))

            boxes = []
            labels = []
            for key in ['stamp_bbx', 'signature_bbx']:
                box = annotation[key]
                if box is None or (isinstance(box, float) and math.isnan(box)):
                    continue
                box = json.loads(box) if isinstance(box, str) else box

                # Ajuster les coordonnées des boîtes englobantes
                x_min = box[0] * original_width
                y_min = box[1] * original_height
                x_max = x_min + (box[2] * original_width)
                y_max = y_min + (box[3] * original_height)

                xmin_final = (x_min / original_width) * self.width
                xmax_final = (x_max / original_width) * self.width
                ymin_final = (y_min / original_height) * self.height
                ymax_final = (y_max / original_height) * self.height

                # Vérifier que toutes les coordonnées sont à l'intérieur de l'image
                xmin_final = min(max(xmin_final, 0), self.width)
                xmax_final = min(max(xmax_final, 0), self.width)
                ymin_final = min(max(ymin_final, 0), self.height)
                ymax_final = min(max(ymax_final, 0), self.height)

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                # Ajouter les étiquettes associées aux boîtes englobantes
                labels.append(1 if key == 'stamp_bbx' else 2)

            # Convertir les listes de boîtes englobantes en tenseur PyTorch
            boxes = torch.tensor(boxes, dtype=torch.float32)

            # Vérifier le type des étiquettes et les convertir en entiers si nécessaire
            if any(isinstance(label, str) for label in labels):
                labels = [self.classes.index(label) for label in labels]
            labels = torch.tensor(labels, dtype=torch.int64)

            # Créer un dictionnaire contenant les informations sur les boîtes englobantes et les étiquettes
            target = {'boxes': boxes, 'labels': labels}
            if self.transforms:
                sample = self.transforms(image = image_resized,  # Utilisez img ici au lieu de original_img
                                         bboxes = target['boxes'],
                                         labels = labels)
                image_resized = sample['image']  # Mettez à jour img avec l'image transformée
                target['boxes'] = torch.Tensor(sample['bboxes'])

            # Vérifier si les boîtes englobantes sont vides ou nulles
            if np.isnan(target['boxes'].numpy()).any() or target['boxes'].shape == torch.Size([0]):
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)


            return image_resized, target

if __name__ == '__main__':
    # Utilisation du dataset avec les transformations d'entraînement
    dataset = InvoiceDataset(
        jsonl_file='/content/drive/MyDrive/requirement samples /Copy of all_data.jsonl',
        image_dir='/content/drive/MyDrive/requirement samples /Copy of 20221109_144858.pdf_1.png',
        # Appliquer la transformation d'entraînement
        width=RESIZE_TO,
        height=RESIZE_TO,
        classes=CLASSES,
        transforms=get_train_transform()
         )

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
     #train_dataset.dataset.transforms = get_train_transform()
     #valid_dataset.dataset.transforms = get_valid_transform()
    print(f"Number of training images: {len(dataset)}")
    train_loader = create_train_loader(train_dataset,) # NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset,) # NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")


    # Définir la liste des noms de classes (remplacez cette liste par vos propres classes si nécessaire)
    class_names = ['__background__', 'stamp', 'signature']
    # Charger deux échantillons de l'ensemble de données

    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt

    def visualize_samples(images, targets, class_names):
        for i, (image, target) in enumerate(zip(images, targets)):
            # If the image is a tensor, convert it to a numpy array and rearrange from CHW to HWC
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()

            # Convert the image to RGB (not necessary if it's already in RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            for box, label in zip(target['boxes'], target['labels']):
                # Create a Rectangle patch
                rectangle = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                          edgecolor='r', facecolor='none')
                plt.gca().add_patch(rectangle)
                plt.gca().text(box[0], box[1] - 2,
                               f'{class_names[label.item()]}',
                               bbox=dict(facecolor='red', alpha=0.5),
                               fontsize=12, color='white')
            plt.title(f'Sample {i+1}')
            plt.axis('off')
        plt.show()



    sample1_idx = 30
    sample2_idx = 50
    img1, target1 = dataset[sample1_idx]
    img2, target2 = dataset[sample2_idx]

    # Visualiser les échantillons
    visualize_samples([img1, img2], [target1, target2], class_names)
