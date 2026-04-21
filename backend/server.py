from collections import OrderedDict

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pickle
from PIL import Image
import io
import base64
import random
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as skimage_ssim


app = FastAPI()


class mymodel(nn.Module):

    def __init__(self):
        super().__init__()

        ###################################################

        self.encoder_hidden_dimension=768
        self.embedding=nn.Linear(self.encoder_hidden_dimension,self.encoder_hidden_dimension)
        self.patchestotal = 196
        self.positions = nn.Parameter(torch.randn(1, self.patchestotal, self.encoder_hidden_dimension))
        
        
        self.encoderhead=12
        self.no_encoderlayers=12
        
        self.encoderlayer=nn.TransformerEncoderLayer(
            d_model=self.encoder_hidden_dimension,
            nhead=self.encoderhead,
            dim_feedforward=self.encoder_hidden_dimension*3,
            batch_first=True
        )
        
        self.encoder=nn.TransformerEncoder(self.encoderlayer,num_layers=self.no_encoderlayers)

##########################################################
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.encoder_hidden_dimension))
        self.decoderhead=6
        self.no_decoderlayer=12
        self.decoder_hidden_dim=384

        self.decoder_embed = nn.Linear(self.encoder_hidden_dimension, self.decoder_hidden_dim)
        
        self.decoderlayer=nn.TransformerEncoderLayer(
            d_model=self.decoder_hidden_dim,
            nhead=self.decoderhead,
            dim_feedforward=self.decoder_hidden_dim*3,
            batch_first=True
        )

        self.decoder=nn.TransformerEncoder(self.decoderlayer,num_layers=self.no_decoderlayer)
        self.decoder_positions = nn.Parameter(torch.randn(1, self.patchestotal, self.decoder_hidden_dim))

        self.output_layer = nn.Linear(self.decoder_hidden_dim, 768)

        self.enco_to_dec_dimension=nn.Linear(self.encoder_hidden_dimension,self.decoder_hidden_dim)



   #*******************************************************************************************************************************************************
    
    def forward(self,image_tensor):
        batch_size=image_tensor.shape[0]
        #print(image_tensor.shape)
        
        patches=patchify(image_tensor)
        patch_embedding=self.embedding(patches)

        patch_embedding=patch_embedding+self.positions

        
        unmask_indices = torch.tensor(random.sample(range(196), 49),device=image_tensor.device)
        unmask_patches = patch_embedding[:,unmask_indices,:]

        encoderoutput=self.encoder(unmask_patches)

        mask_indices=[]
        for i in range(196):
            if i not in unmask_indices:
                mask_indices.append(i)

        mask_indices_tensor=torch.tensor(mask_indices,device=image_tensor.device)
        
        mask_tokens = self.mask_token.expand(batch_size, 196-49, self.encoder_hidden_dimension)

        decoder_input = torch.zeros(batch_size, 196, 768, device=image_tensor.device)
        decoder_input[:, unmask_indices, :] = encoderoutput
        
        encoderoutput=self.enco_to_dec_dimension(encoderoutput)
        decoder_input[:, mask_indices, :] = mask_tokens

        
        decoder_input=self.decoder_embed(decoder_input)
        #print(encoderoutput.shape)
        #print(decoder_input.shape)
        decoder_input=decoder_input+self.decoder_positions
        decoder_output = self.decoder(
            decoder_input,  
            #memory=encoderoutput  
        )

        reconstruction = self.output_layer(decoder_output)  # (B, N, patch_dim)
        #print(reconstruction.shape)
        return reconstruction,mask_indices_tensor, unmask_indices
        

        
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mymodel = mymodel().to(device)

checkpoint = torch.load("model5.pth", map_location=device)
# Remove DataParallel prefix if exists
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

mymodel.load_state_dict(new_state_dict)
mymodel.eval()

# -----------------------------
def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def resize_image(img: Image.Image) -> torch.Tensor:
    """Resize image to 224x224 and convert to tensor"""
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img)  # HWC
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # CHW
    img = img.unsqueeze(0)  # add batch dimension
    return img  # shape: [1, 3, 224, 224]


def patchify(img: torch.Tensor) -> torch.Tensor:
    """Convert image tensor to patches"""
    batch, channels, height, width = img.shape
    patch_size = 16
    patchh = height // patch_size
    patchw = width // patch_size

    x = img.reshape(batch, channels, patchh, patch_size, patchw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    patches = x.reshape(batch, patchh * patchw, channels * patch_size * patch_size)
    return patches  # shape: [batch, num_patches, patch_dim]


def unpatchify(patches: torch.Tensor, img_size=224, patch_size=16) -> torch.Tensor:
    """Convert patches back to image tensor"""
    batch, num_patches, patch_dim = patches.shape
    channels = 3  # Fixed for RGB
    num_patches_h = img_size // patch_size  # 14
    num_patches_w = img_size // patch_size  # 14
    
    # Reshape to (B, H/patch, W/patch, C, patch, patch)
    patches = patches.reshape(batch, num_patches_h, num_patches_w, 
                              channels, patch_size, patch_size)
    
    # Permute to (B, C, H/patch, patch, W/patch, patch)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    
    # Reshape to final image (B, C, H, W)
    img = patches.reshape(batch, channels, img_size, img_size)
    return img


def _tensor_to_hwc_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    return np.round(image * 255.0).astype(np.uint8)


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor):
    try:
        original = torch.clamp(original.detach(), 0.0, 1.0)
        reconstructed = torch.clamp(reconstructed.detach(), 0.0, 1.0)
        mse = torch.mean((original - reconstructed) ** 2).item()
        if mse <= 1e-12:
            return 100.0
        return float(10.0 * np.log10(1.0 / mse))
    except Exception:
        return None


def compute_ssim(original: torch.Tensor, reconstructed: torch.Tensor):
    if skimage_ssim is None:
        return None

    try:
        original_uint8 = _tensor_to_hwc_uint8(original)
        reconstructed_uint8 = _tensor_to_hwc_uint8(reconstructed)
        score = skimage_ssim(original_uint8, reconstructed_uint8, channel_axis=2, data_range=255)
        return float(score)
    except Exception:
        return None

# -----------------------------
# FastAPI endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # -----------------------------
    # Read uploaded image
    # -----------------------------
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # -----------------------------
    # Resize & convert to tensor
    # -----------------------------
    img_tensor = resize_image(image).to(device)  # shape: [1, 3, 224, 224]

    # -----------------------------
    # Patchify
    # -----------------------------
    original_patches = patchify(img_tensor)  # [1, 196, 768]

    # -----------------------------
    # Run model
    # -----------------------------
    with torch.no_grad():
        reconstructed_patches, maskedpatcheslist, unmask_indices = mymodel(img_tensor)

    # # -----------------------------
    # # Build masked & generated images
    # # -----------------------------
    # # Masked patches
    # masked_output = results[:, maskedpatcheslist, :]
    # masked_patches = original_patches.clone()
    # masked_patches[:, maskedpatcheslist, :] = 0  # zero-out masked patches

    # # Convert patches back to images
    # masked_img_tensor = unpatchify(masked_patches)  # [1, 3, H, W]
    # generated_img_tensor = unpatchify(results)      # [1, 3, H, W]

    # # Convert tensors to PIL


    #     # 1. Original image (unmodified)
    # original_img_tensor = img_tensor  # [1, 3, H, W]
    
    # 2. Masked image (masked patches set to 0/gray)
    masked_patches = original_patches.clone()
    masked_patches[:, maskedpatcheslist, :] = 0  # zero-out masked patches
    masked_img_tensor = unpatchify(masked_patches)
    
    # 3. Reconstructed image (using model's full output)
    #reconstructed_img_tensor = unpatchify(reconstructed_patches)
    
    # 4. (Optional) Reconstruction with original unmasked + predicted masked
    combined_patches = original_patches.clone()
    combined_patches[:, maskedpatcheslist, :] = reconstructed_patches[:, maskedpatcheslist, :]
    combined_img_tensor = unpatchify(combined_patches)

    # Quantitative evaluation for full-image reconstruction quality
    psnr_value = compute_psnr(img_tensor, combined_img_tensor)
    ssim_value = compute_ssim(img_tensor, combined_img_tensor)

    full_recon_img_tensor = unpatchify(reconstructed_patches)
    def tensor_to_pil(tensor):
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC
    
    # IMPORTANT: Clip to [0,1] before scaling to 0-255
        tensor = np.clip(tensor, 0, 1)
    
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)

    masked_img = tensor_to_pil(masked_img_tensor)
    #combined_img = tensor_to_pil(combined_img_tensor)
    #full_recon_img = tensor_to_pil(full_recon_img_tensor)
    generated_img = tensor_to_pil(combined_img_tensor)
    #original_img = tensor_to_pil(img_tensor)
    # -----------------------------
    # Return as base64
    # -----------------------------
    masked_b64 = image_to_base64(masked_img)
    generated_b64 = image_to_base64(generated_img)

    return JSONResponse({
        "masked": masked_b64,
        "generated": generated_b64,
        "psnr": round(psnr_value, 2) if psnr_value is not None else None,
        "ssim": round(ssim_value, 2) if ssim_value is not None else None
    })