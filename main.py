import torch  # unchanged
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet_parts import UNet
from PIL import Image #assisted
import numpy as np #assisted
import os  #assisted
import random #assisted

#Dataset class for images and masks
class CellDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, masks_path, image_size=(256, 256)):
# Check folder exists
        if not os.path.isdir(images_path) or not os.path.isdir(masks_path):
            raise FileNotFoundError("Images or masks folder not found")

# Load all image and mask paths
        self.images = sorted([os.path.join(images_path, str(f)) for f in os.listdir(images_path)
                              if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])
        self.masks = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path)
                             if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])

        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks must match")

# Resize images for memory safety
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
# Load and process image
        image = Image.open(self.images[index]).convert("RGB").resize(self.image_size)
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2,0,1) / 255.0

# Load and process mask
        mask_img = Image.open(self.masks[index]).convert("L").resize(self.image_size)
        mask_tensor = torch.tensor(np.array(mask_img), dtype=torch.float32).unsqueeze(0) / 255.0

        return image, mask_tensor, os.path.basename(self.images[index])

#Function to save predicted segmentation
def save_segmentation(pred_tensor_seg, save_path, filenamez): #filenamez used to aviod outscope
    os.makedirs(save_path, exist_ok=True)
    pred_mask = torch.sigmoid(pred_tensor_seg).squeeze(0).cpu().numpy() * 255
    pred_mask = pred_mask.astype(np.uint8)
    output = Image.fromarray(pred_mask)
    output.save(os.path.join(save_path, filenamez))

if __name__ == '__main__':
#Hyperparameters and paths
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 2
    EPOCHS = 2
    DATA_PATH_IMAGES = '/Users/nickjohnson/Desktop/IMG Folder/images'
    DATA_PATH_MASKS = '/Users/nickjohnson/Desktop/IMG Folder/masks'
    MODEL_SAVE_PATH = '/Users/nickjohnson/Desktop/IMG Folder/test_model.pth'
    SEG_SAVE_PATH = '/Users/nickjohnson/Desktop/IMG Folder/IMG File'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CellDataset(DATA_PATH_IMAGES, DATA_PATH_MASKS)  #assisted

    generator = torch.Generator().manual_seed(42)
    dataset_size = len(train_dataset)
    val_size = max(1, int(0.2 * dataset_size))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

#Model, optimizer, loss
    model = UNet(in_ch=3, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

#Training loop
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        idx = -1 #this initializes idx before the loop in the case the loop does not run
        for idx, (img, mask, _) in enumerate(tqdm(train_dataloader)):  #assisted
            img = img.float().to(device)
            mask = mask.float().to(device)

            optimizer.zero_grad()
            y_prediction = model(img)
            loss = criterion(y_prediction, mask)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        train_loss = train_running_loss / (idx + 1)

#Validation loop
        model.eval()
        val_running_loss = 0
        all_val_samples = []  #collect samples for random segmentation
        with torch.no_grad():
            for val_idx, (img, mask, filenames) in enumerate(tqdm(val_dataloader)):  # unpack images, masks, filenames
                val_img = img.float().to(device)
                val_mask = mask.float().to(device)

                y_prediction = model(img)
                loss = criterion(y_prediction, mask)
                val_running_loss += loss.item()

#Collect all validation samples
                for i in range(img.size(0)):
                    all_val_samples.append((y_prediction[i], filenames[i]))

        val_loss = val_running_loss / (val_idx + 1)

#Save 2–5 random segmented images per epoch
        random_samples = random.sample(all_val_samples, min(5, len(all_val_samples)))  # changed to save a few examples
        for pred_tensor, filename in random_samples:
            save_segmentation(pred_tensor, SEG_SAVE_PATH, filename)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-"*30)

#Save final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

#Debug one batch
    batch = next(iter(val_dataloader))
    print("TYPE:", type(batch))
    print("LENGTH:", len(batch))
    for i, b in enumerate(batch):
        print(f"Element {i} type:", type(b))