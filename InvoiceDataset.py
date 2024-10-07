{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyOC8eGSDGZEFTFu6gi6VpDo",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/omar80-coder/Document-validation-Stamp-and-Signature-detection-using-Deep-Learning-/blob/main/InvoiceDataset.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fkolRGV4ZEym"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "24cefe13-dcb2-417c-96df-984ad6181bd9"
      },
      "outputs": [],
      "source": [
        "import albumentations as A\n",
        "import cv2\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from albumentations.pytorch import ToTensorV2\n",
        "\n",
        "\n",
        "plt.style.use('ggplot')\n",
        "\n",
        "# This class keeps track of the training and validation loss values\n",
        "# and helps to get the average for each epoch as well.\n",
        "class Averager:\n",
        "    def __init__(self):\n",
        "        self.current_total = 0.0\n",
        "        self.iterations = 0.0\n",
        "\n",
        "    def send(self, value):\n",
        "        self.current_total += value\n",
        "        self.iterations += 1\n",
        "\n",
        "    @property\n",
        "    def value(self):\n",
        "        if self.iterations == 0:\n",
        "            return 0\n",
        "        else:\n",
        "            return 1.0 * self.current_total / self.iterations\n",
        "\n",
        "    def reset(self):\n",
        "        self.current_total = 0.0\n",
        "        self.iterations = 0.0\n",
        "\n",
        "class SaveBestModel:\n",
        "    \"\"\"\n",
        "    Class to save the best model during training based on the validation mAP.\n",
        "    \"\"\"\n",
        "    def __init__(self, best_valid_map=float('-inf')):\n",
        "        \"\"\"\n",
        "        Initialize the SaveBestModel object with the best validation mAP observed.\n",
        "\n",
        "        Parameters:\n",
        "        best_valid_map (float): The best validation mAP observed so far.\n",
        "        \"\"\"\n",
        "        self.best_valid_map = best_valid_map\n",
        "\n",
        "    def __call__(self, model, current_valid_map, epoch, OUT_DIR):\n",
        "        \"\"\"\n",
        "        If the current validation mAP is higher than the previously observed best, save the model.\n",
        "\n",
        "        Parameters:\n",
        "        model (torch.nn.Module): The model to save.\n",
        "        current_valid_map (float): The current epoch's validation mAP.\n",
        "        epoch (int): The current epoch number.\n",
        "        OUT_DIR (str): Directory to save the best model.\n",
        "        \"\"\"\n",
        "        if current_valid_map > self.best_valid_map:\n",
        "            self.best_valid_map = current_valid_map\n",
        "            print(f\"\\nBEST VALIDATION mAP: {self.best_valid_map}\")\n",
        "            print(f\"SAVING BEST MODEL AT EPOCH: {epoch+1}\\n\")\n",
        "            path = f\"{OUT_DIR}/best_model.pth\"\n",
        "            # Sauvegarder le dictionnaire de l'état du modèle\n",
        "            torch.save({\n",
        "                'epoch': epoch + 1,\n",
        "                'model_state_dict': model.state_dict(),\n",
        "            }, path)\n",
        "\n",
        "            # Enregistrer l'artefact dans MLflow\n",
        "            #mlflow.log_artifact(path, \"best_model\")\n",
        "def collate_fn(batch):\n",
        "    \"\"\"\n",
        "    To handle the data loading as different images may have different number\n",
        "    of objects and to handle varying size tensors as well.\n",
        "    \"\"\"\n",
        "    return tuple(zip(*batch))\n",
        "def create_train_loader(train_dataset, num_workers=0):\n",
        "    train_loader = DataLoader(\n",
        "        train_dataset,\n",
        "        batch_size=BATCH_SIZE,\n",
        "        shuffle=True,\n",
        "        num_workers=num_workers,\n",
        "        collate_fn=collate_fn,\n",
        "        drop_last=True\n",
        "    )\n",
        "    return train_loader\n",
        "def create_valid_loader(valid_dataset, num_workers=0):\n",
        "    valid_loader = DataLoader(\n",
        "        valid_dataset,\n",
        "        batch_size=BATCH_SIZE,\n",
        "        shuffle=False,\n",
        "        num_workers=num_workers,\n",
        "        collate_fn=collate_fn,\n",
        "        drop_last=True\n",
        "    )\n",
        "    return valid_loader\n",
        "\n",
        "# Define the training tranforms.\n",
        "import albumentations as A\n",
        "def get_train_transform():\n",
        "    return A.Compose([\n",
        "        A.Rotate(limit=10, p=0.5),\n",
        "        A.ColorJitter(p=0.4),\n",
        "        A.MultiplicativeNoise(multiplier=(0.5, 1), per_channel=True, p=0.3),\n",
        "        ToTensorV2(p=1.0),\n",
        "    ], bbox_params={\n",
        "        'format': 'pascal_voc',\n",
        "        'label_fields': ['labels']\n",
        "    })\n",
        "\n",
        "# Define the validation transforms.\n",
        "def get_valid_transform():\n",
        "    return A.Compose([\n",
        "        ToTensorV2(p=1.0),\n",
        "    ], bbox_params={\n",
        "        'format': 'pascal_voc',\n",
        "        'label_fields': ['labels']\n",
        "    })\n",
        "\n",
        "\n",
        "def show_tranformed_image(train_loader):\n",
        "    \"\"\"\n",
        "    This function shows the transformed images from the `train_loader`.\n",
        "    Helps to check whether the tranformed images along with the corresponding\n",
        "    labels are correct or not.\n",
        "    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.\n",
        "    \"\"\"\n",
        "    if len(train_loader) > 0:\n",
        "        for i in range(1):\n",
        "            images, targets = next(iter(train_loader))\n",
        "            images = list(image.to(DEVICE) for image in images)\n",
        "            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]\n",
        "            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)\n",
        "            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)\n",
        "            sample = images[i].permute(1, 2, 0).cpu().numpy()\n",
        "            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)\n",
        "            for box_num, box in enumerate(boxes):\n",
        "                cv2.rectangle(sample,\n",
        "                            (box[0], box[1]),\n",
        "                            (box[2], box[3]),\n",
        "                            (0, 0, 255), 2)\n",
        "                cv2.putText(sample, CLASSES[labels[box_num]],\n",
        "                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX,\n",
        "                            1.0, (0, 0, 255), 2)\n",
        "            cv2.imshow('Transformed image', sample)\n",
        "            cv2.waitKey(0)\n",
        "            cv2.destroyAllWindows()\n",
        "\n",
        "\n",
        "def save_model(epoch, model, optimizerr,scheduler, train_loss_list, validation_loss_list, map_50_list, map_list, OUT_DIR='outputs'):\n",
        "    \"\"\"\n",
        "    Function to save the trained model till the current epoch, or whenever called.\n",
        "    Additionally, save the training and validation loss and mAP history for continuity in tracking the model's performance.\n",
        "\n",
        "    Parameters:\n",
        "    epoch (int): Current epoch number.\n",
        "    model (torch.nn.Module): Model to be saved.\n",
        "    optimizer (torch.optim.Optimizer): Optimizer to be saved.\n",
        "    scheduler (torch.optim.lr_scheduler): Scheduler to be saved.\n",
        "    train_loss_list (list): List of training loss values.\n",
        "    validation_loss_list (list): List of validation loss values.\n",
        "    map_50_list (list): List of mAP@0.50 values.\n",
        "    map_list (list): List of mean average precision values.\n",
        "    OUT_DIR (str): Directory to save the model checkpoint.\n",
        "    \"\"\"\n",
        "    path = f'{OUT_DIR}/last_model.pth'  # Define path for clarity and troubleshooting\n",
        "    torch.save({\n",
        "        'epoch': epoch,\n",
        "        'model_state_dict': model.state_dict(),\n",
        "        'optimizer_state_dict': optimizer.state_dict(),\n",
        "        'scheduler_state_dict': scheduler.state_dict(),  # Ensure scheduler is passed and saved\n",
        "        'train_loss_list': train_loss_list,\n",
        "        'validation_loss_list': validation_loss_list,\n",
        "        'map_50_list': map_50_list,\n",
        "        'map_list': map_list,\n",
        "    }, path)\n",
        "     #mlflow.log_artifact(path, \"model_checkpoints\")\n",
        "\n",
        "    print(f'Model and metrics saved for epoch {epoch}')  # Adjusted to display current epoch\n",
        "\n",
        "\n",
        "\n",
        "def save_loss_plot(OUT_DIR, train_loss_list, validation_loss_list, x_label='Epochs', y_label='Loss', save_name='loss_plot'):\n",
        "    \"\"\"\n",
        "    Function to save the training and validation loss graphs.\n",
        "\n",
        "    Parameters:\n",
        "    OUT_DIR (str): Directory to save the plot.\n",
        "    train_loss_list (list): List of training loss values.\n",
        "    validation_loss_list (list): List of validation loss values.\n",
        "    x_label (str): Label for the x-axis.\n",
        "    y_label (str): Label for the y-axis.\n",
        "    save_name (str): Filename for the saved plot.\n",
        "    \"\"\"\n",
        "    plt.figure(figsize=(10, 7))\n",
        "    plt.plot(train_loss_list, label='Training Loss', color='blue')\n",
        "    plt.plot(validation_loss_list, label='Validation Loss', color='red')\n",
        "    plt.xlabel(x_label)\n",
        "    plt.ylabel(y_label)\n",
        "    plt.title('Training and Validation Loss')\n",
        "    plt.legend()\n",
        "    plt.savefig(f\"{OUT_DIR}/{save_name}.png\")\n",
        "    plt.close()\n",
        "    print('SAVING LOSS PLOTS COMPLETE...')\n",
        "\n",
        "def save_mAP(OUT_DIR, map_05, map):\n",
        "    \"\"\"\n",
        "    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.\n",
        "    :param OUT_DIR: Path to save the graphs.\n",
        "    :param map_05: List containing mAP values at 0.5 IoU (expected as list of tensors or floats).\n",
        "    :param map: List containing mAP values at 0.5:0.95 IoU (expected as list of tensors or floats).\n",
        "    \"\"\"\n",
        "    if len(map_05) > 0 and isinstance(map_05[0], torch.Tensor):\n",
        "        map_05 = [m.cpu().item() for m in map_05]  # Appliquer .cpu() à chaque tenseur\n",
        "    if len(map) > 0 and isinstance(map[0], torch.Tensor):\n",
        "        map = [m.cpu().item() for m in map]\n",
        "    # Création du graphique\n",
        "    figure = plt.figure(figsize=(10, 7), num=1, clear=True)\n",
        "    ax = figure.add_subplot()\n",
        "    ax.plot(\n",
        "        map_05, color='tab:orange', linestyle='-',\n",
        "        label='mAP@0.5'\n",
        "    )\n",
        "    ax.plot(\n",
        "        map, color='tab:red', linestyle='-',\n",
        "        label='mAP@0.5:0.95'\n",
        "    )\n",
        "    ax.set_xlabel('Epochs')\n",
        "    ax.set_ylabel('mAP')\n",
        "    ax.legend()\n",
        "    plt.title('mAP over Epochs')\n",
        "    plt.grid(True)\n",
        "    figure.savefig(f\"{OUT_DIR}/map.png\")\n",
        "    plt.close(figure)  # Fermez le graphique pour libérer de la mémoire\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "515017c3-faa0-45cc-8924-ac874265b5d9",
        "outputId": "bfc7dbe5-6c16-4fa2-a02c-ec3b352ce0f4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 443
        },
        "collapsed": true
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Number of training images: 381\n",
            "Number of training samples: 304\n",
            "Number of validation samples: 77\n",
            "\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "NotADirectoryError",
          "evalue": "[Errno 20] Not a directory: '/content/drive/MyDrive/requirement samples /Copy of 20221109_144858.pdf_1.png/20231217_190133_merged-43.pdf_1.png'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNotADirectoryError\u001b[0m                        Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-10-876c7988d69d>\u001b[0m in \u001b[0;36m<cell line: 86>\u001b[0;34m()\u001b[0m\n\u001b[1;32m    144\u001b[0m     \u001b[0msample1_idx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m30\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    145\u001b[0m     \u001b[0msample2_idx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m50\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 146\u001b[0;31m     \u001b[0mimg1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget1\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdataset\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0msample1_idx\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    147\u001b[0m     \u001b[0mimg2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget2\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdataset\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0msample2_idx\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    148\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m<ipython-input-10-876c7988d69d>\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, idx)\u001b[0m\n\u001b[1;32m     22\u001b[0m             \u001b[0mannotation\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mannotations\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0midx\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     23\u001b[0m             \u001b[0mimg_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimage_dir\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mannotation\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'filename'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 24\u001b[0;31m             \u001b[0moriginal_img\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mImage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimg_path\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconvert\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"RGB\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     25\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     26\u001b[0m             \u001b[0;31m# Obtenir les dimensions originales de l'image\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/PIL/Image.py\u001b[0m in \u001b[0;36mopen\u001b[0;34m(fp, mode, formats)\u001b[0m\n\u001b[1;32m   3225\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3226\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mfilename\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3227\u001b[0;31m         \u001b[0mfp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbuiltins\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"rb\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3228\u001b[0m         \u001b[0mexclusive_fp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3229\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNotADirectoryError\u001b[0m: [Errno 20] Not a directory: '/content/drive/MyDrive/requirement samples /Copy of 20221109_144858.pdf_1.png/20231217_190133_merged-43.pdf_1.png'"
          ]
        }
      ],
      "source": [
        "from torch.utils.data import DataLoader, random_split\n",
        "from torchvision import transforms\n",
        "from PIL import Image\n",
        "import json\n",
        "import os\n",
        "import math\n",
        "import cv2\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "import numpy as np\n",
        "class InvoiceDataset(torch.utils.data.Dataset):\n",
        "    def __init__(self, jsonl_file, image_dir, width, height, classes, transforms=None):\n",
        "        self.annotations = [json.loads(line) for line in open(jsonl_file, 'r')]\n",
        "        self.image_dir = image_dir\n",
        "        self.height = height\n",
        "        self.width = width\n",
        "        self.classes = classes\n",
        "        self.transforms = transforms\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.annotations)\n",
        "    def __getitem__(self, idx):\n",
        "            annotation = self.annotations[idx]\n",
        "            img_path = os.path.join(self.image_dir, annotation['filename'])\n",
        "            original_img = Image.open(img_path).convert(\"RGB\")\n",
        "\n",
        "            # Obtenir les dimensions originales de l'image\n",
        "            original_width, original_height = original_img.size\n",
        "            image = cv2.imread(img_path)\n",
        "            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR en RGB\n",
        "            image = image.astype(np.float32) / 255.0  # Convertir en float32 et normaliser\n",
        "            image_resized = cv2.resize(image, (self.width, self.height))\n",
        "\n",
        "            boxes = []\n",
        "            labels = []\n",
        "            for key in ['stamp_bbx', 'signature_bbx']:\n",
        "                box = annotation[key]\n",
        "                if box is None or (isinstance(box, float) and math.isnan(box)):\n",
        "                    continue\n",
        "                box = json.loads(box) if isinstance(box, str) else box\n",
        "\n",
        "                # Ajuster les coordonnées des boîtes englobantes\n",
        "                x_min = box[0] * original_width\n",
        "                y_min = box[1] * original_height\n",
        "                x_max = x_min + (box[2] * original_width)\n",
        "                y_max = y_min + (box[3] * original_height)\n",
        "\n",
        "                xmin_final = (x_min / original_width) * self.width\n",
        "                xmax_final = (x_max / original_width) * self.width\n",
        "                ymin_final = (y_min / original_height) * self.height\n",
        "                ymax_final = (y_max / original_height) * self.height\n",
        "\n",
        "                # Vérifier que toutes les coordonnées sont à l'intérieur de l'image\n",
        "                xmin_final = min(max(xmin_final, 0), self.width)\n",
        "                xmax_final = min(max(xmax_final, 0), self.width)\n",
        "                ymin_final = min(max(ymin_final, 0), self.height)\n",
        "                ymax_final = min(max(ymax_final, 0), self.height)\n",
        "\n",
        "                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])\n",
        "                # Ajouter les étiquettes associées aux boîtes englobantes\n",
        "                labels.append(1 if key == 'stamp_bbx' else 2)\n",
        "\n",
        "            # Convertir les listes de boîtes englobantes en tenseur PyTorch\n",
        "            boxes = torch.tensor(boxes, dtype=torch.float32)\n",
        "\n",
        "            # Vérifier le type des étiquettes et les convertir en entiers si nécessaire\n",
        "            if any(isinstance(label, str) for label in labels):\n",
        "                labels = [self.classes.index(label) for label in labels]\n",
        "            labels = torch.tensor(labels, dtype=torch.int64)\n",
        "\n",
        "            # Créer un dictionnaire contenant les informations sur les boîtes englobantes et les étiquettes\n",
        "            target = {'boxes': boxes, 'labels': labels}\n",
        "            if self.transforms:\n",
        "                sample = self.transforms(image = image_resized,  # Utilisez img ici au lieu de original_img\n",
        "                                         bboxes = target['boxes'],\n",
        "                                         labels = labels)\n",
        "                image_resized = sample['image']  # Mettez à jour img avec l'image transformée\n",
        "                target['boxes'] = torch.Tensor(sample['bboxes'])\n",
        "\n",
        "            # Vérifier si les boîtes englobantes sont vides ou nulles\n",
        "            if np.isnan(target['boxes'].numpy()).any() or target['boxes'].shape == torch.Size([0]):\n",
        "                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)\n",
        "\n",
        "\n",
        "            return image_resized, target\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    # Utilisation du dataset avec les transformations d'entraînement\n",
        "    dataset = InvoiceDataset(\n",
        "        jsonl_file='/content/drive/MyDrive/requirement samples /Copy of all_data.jsonl',\n",
        "        image_dir='/content/drive/MyDrive/requirement samples /Copy of 20221109_144858.pdf_1.png',\n",
        "        # Appliquer la transformation d'entraînement\n",
        "        width=RESIZE_TO,\n",
        "        height=RESIZE_TO,\n",
        "        classes=CLASSES,\n",
        "        transforms=get_train_transform()\n",
        "         )\n",
        "\n",
        "    train_size = int(0.8 * len(dataset))\n",
        "    valid_size = len(dataset) - train_size\n",
        "    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])\n",
        "     #train_dataset.dataset.transforms = get_train_transform()\n",
        "     #valid_dataset.dataset.transforms = get_valid_transform()\n",
        "    print(f\"Number of training images: {len(dataset)}\")\n",
        "    train_loader = create_train_loader(train_dataset,) # NUM_WORKERS)\n",
        "    valid_loader = create_valid_loader(valid_dataset,) # NUM_WORKERS)\n",
        "    print(f\"Number of training samples: {len(train_dataset)}\")\n",
        "    print(f\"Number of validation samples: {len(valid_dataset)}\\n\")\n",
        "\n",
        "\n",
        "    # Définir la liste des noms de classes (remplacez cette liste par vos propres classes si nécessaire)\n",
        "    class_names = ['__background__', 'stamp', 'signature']\n",
        "    # Charger deux échantillons de l'ensemble de données\n",
        "\n",
        "    import matplotlib.pyplot as plt\n",
        "\n",
        "    import matplotlib.pyplot as plt\n",
        "\n",
        "    def visualize_samples(images, targets, class_names):\n",
        "        for i, (image, target) in enumerate(zip(images, targets)):\n",
        "            # If the image is a tensor, convert it to a numpy array and rearrange from CHW to HWC\n",
        "            if isinstance(image, torch.Tensor):\n",
        "                image = image.permute(1, 2, 0).cpu().numpy()\n",
        "\n",
        "            # Convert the image to RGB (not necessary if it's already in RGB)\n",
        "            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
        "\n",
        "            plt.figure(figsize=(8, 8))\n",
        "            plt.imshow(image)\n",
        "            for box, label in zip(target['boxes'], target['labels']):\n",
        "                # Create a Rectangle patch\n",
        "                rectangle = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],\n",
        "                                          edgecolor='r', facecolor='none')\n",
        "                plt.gca().add_patch(rectangle)\n",
        "                plt.gca().text(box[0], box[1] - 2,\n",
        "                               f'{class_names[label.item()]}',\n",
        "                               bbox=dict(facecolor='red', alpha=0.5),\n",
        "                               fontsize=12, color='white')\n",
        "            plt.title(f'Sample {i+1}')\n",
        "            plt.axis('off')\n",
        "        plt.show()\n",
        "\n",
        "\n",
        "\n",
        "    sample1_idx = 30\n",
        "    sample2_idx = 50\n",
        "    img1, target1 = dataset[sample1_idx]\n",
        "    img2, target2 = dataset[sample2_idx]\n",
        "\n",
        "    # Visualiser les échantillons\n",
        "    visualize_samples([img1, img2], [target1, target2], class_names)\n"
      ]
    }
  ]
}