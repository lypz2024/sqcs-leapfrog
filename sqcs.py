import os
import argparse
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from torchvision.models import squeezenet1_1
import warnings
import time

warnings.filterwarnings('ignore')

class CosineSimilarity:
    """Class tasked with comparing similarity between two images"""
    
    def __init__(self, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def model(self):
        """Instantiates the feature extracting model using SqueezeNet v1.1
        
        Returns
        -------
        SqueezeNet model object
        """
        model = squeezenet1_1(pretrained=True)
        model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())  
        model = model.to(self.device)
        model.eval()
        return model

    def process_image(self, image_path):
        """Processing images

        Parameters
        ----------
        image_path : str

        Returns
        -------
        Processed image : torch.Tensor
        """
        img = Image.open(image_path).convert('RGB')
        transformations = tr.Compose([tr.Resize((224, 224)),  
                                      tr.ToTensor(),
                                      tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transformations(img).float()
        img = img.unsqueeze_(0)
        img = img.to(self.device)
        return img

    def process_images_in_batch(self, image_paths):
        """Processing images in batches

        Parameters
        ----------
        image_paths : list of str

        Returns
        -------
        Processed images : torch.Tensor
        """
        images = []
        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')
            transformations = tr.Compose([tr.Resize((224, 224)),  
                                          tr.ToTensor(),
                                          tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            img = transformations(img).float()
            images.append(img)

        images = torch.stack(images)
        images = images.to(self.device)
        return images

    def compute_scores(self, img_batch, gen_img_batch):
        """Computes cosine similarity between batches of real and generated images."""
        model = self.model()
        with torch.no_grad():
            emb_batch_real = model(img_batch).detach().cpu()
            emb_batch_gen = model(gen_img_batch).detach().cpu()
            scores = torch.nn.functional.cosine_similarity(emb_batch_real.unsqueeze(1), emb_batch_gen.unsqueeze(0), dim=2)
        return scores.numpy()

def load_images(image_dir, num_images=None):
    """Load images from a directory."""
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files = image_files[:num_images]
    images = [os.path.join(image_dir, img) for img in image_files]
    return images if images else None

def compute_mean_cosine_similarity(real_image_dir, gen_image_dir, batch_size):
    start_time = time.time()  # Start time

    # Create an instance of the CosineSimilarity class
    cs = CosineSimilarity()

    # Load and process real images from the directory
    real_images = load_images(real_image_dir)
    if real_images is None:
        print("No real images found in directory:", real_image_dir)
        return None

    # Load and process generated images from the directory
    gen_images = load_images(gen_image_dir)
    if gen_images is None:
        print("No generated images found in directory:", gen_image_dir)
        return None

    # Process real images
    real_images_processed = cs.process_images_in_batch(real_images)

    # Process generated images
    gen_images_processed = cs.process_images_in_batch(gen_images)

    # Compute cosine similarity scores between all real and generated images
    scores = cs.compute_scores(real_images_processed, gen_images_processed)

    # Calculate mean cosine similarity score for all images
    mean_score = np.mean(scores)
    end_time = time.time()  # End time
    time_taken = end_time - start_time

    print(f"Mean Cosine Similarity Score for all images:", round(mean_score, 4))
    print(f"Time taken for computing scores:", round(time_taken, 4), "seconds")
    return mean_score

def main(real_image_dir, gen_image_dir, batch_size):
    compute_mean_cosine_similarity(real_image_dir, gen_image_dir, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cosine similarity between real images and generated images.")
    parser.add_argument('real_image_dir', type=str, help="Directory containing real images")
    parser.add_argument('gen_image_dir', type=str, help="Directory containing generated images")
    parser.add_argument('--batch_size', type=int, default=500, help="Batch size for processing images")
    args = parser.parse_args()

    main(args.real_image_dir, args.gen_image_dir, args.batch_size)