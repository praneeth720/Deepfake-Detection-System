from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTImageProcessor
import torch


def load_image_models():
    print("Loading image models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Smaller CLIP model (FAST & STABLE)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_model.to(device)
    clip_model.eval()

    # Smaller ViT model
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    vit_model.to(device)
    vit_model.eval()

    print("Image models loaded successfully")

    return clip_model, clip_processor, vit_model, vit_processor