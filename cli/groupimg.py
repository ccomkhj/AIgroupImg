import os
import shutil
import glob
import math
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

# Advanced feature extraction and dimensionality reduction imports
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import timm for advanced model support (conditional import)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Advanced models will not be accessible.")
    print("Install with: pip install timm")

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore')

# Check for CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature extraction models
def get_feature_extractor(model_name, img_size=224):
    """
    Initialize and return a feature extraction model.
    """
    if not TIMM_AVAILABLE and model_name != 'histogram':
        print(f"Warning: timm library not available, falling back to histogram method.")
        return None
        
    if model_name == 'histogram':
        return None
    elif model_name in ['vit', 'swin', 'efficientnetv2']:
        model_map = {
            'vit': 'vit_base_patch16_224.augreg_in21k',
            'swin': 'swin_base_patch4_window7_224.ms_in22k',
            'efficientnetv2': 'tf_efficientnetv2_b3.in21k'
        }
        
        # Create model and remove classification head
        model = timm.create_model(model_map[model_name], pretrained=True, num_classes=0)
        model.eval()
        model = model.to(DEVICE)
        
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Transform for deep learning models
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class K_means:

  def __init__(self, k=3, size=False, resample=32, feature_method='histogram', dim_reduction=None, dim_components=50):
    self.k = k
    self.cluster = []
    self.data = []
    self.end = []
    self.i = 0
    self.size = size
    self.resample = resample
    
    # Advanced options
    self.feature_method = feature_method
    self.dim_reduction = dim_reduction
    self.dim_components = dim_components
    
    # Initialize feature extractor if needed
    self.model = get_feature_extractor(feature_method)
    
    # Initialize transform for deep learning models
    if feature_method != 'histogram':
        self.transform = get_transform()
    
    # Dimensionality reduction object (will be initialized after features are extracted)
    self.dim_reducer = None

  def manhattan_distance(self,x1,x2):
    s = 0.0
    for i in range(len(x1)):
      s += abs( float(x1[i]) - float(x2[i]) )
    return s

  def euclidian_distance(self,x1,x2):
    s = 0.0
    for i in range(len(x1)):
      s += math.sqrt((float(x1[i]) - float(x2[i])) ** 2)
    return s

  def read_image(self,im):
    if self.i >= self.k :
      self.i = 0
    try:
      img = Image.open(im).convert('RGB')  # Ensure RGB mode for all models
      osize = img.size
      
      # Feature extraction based on selected method
      if self.feature_method == 'histogram':
        # Original histogram-based method
        img.thumbnail((self.resample,self.resample))
        v = [float(p)/float(img.size[0]*img.size[1])*100 for p in np.histogram(np.asarray(img))[0]]
        if self.size:
          v += [osize[0], osize[1]]
      else:
        # Deep learning model-based feature extraction
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
          features = self.model(img_tensor)
        
        # Convert to numpy and flatten
        v = features.squeeze().cpu().numpy().flatten()
        
        # Add size features if requested
        if self.size:
          # Normalize size values to have similar scale with features
          norm_width = float(osize[0]) / 1000.0
          norm_height = float(osize[1]) / 1000.0
          v = np.append(v, [norm_width, norm_height])
      
      pbar.update(1)
      i = self.i
      self.i += 1
      return [i, v, im]
    except Exception as e:
      print("Error reading ",im,e)
      return [None, None, None]


  def generate_k_means(self):
    final_mean = []
    for c in range(self.k):
      partial_mean = []
      for i in range(len(self.data[0])):
        s = 0.0
        t = 0
        for j in range(len(self.data)):
          if self.cluster[j] == c :
            s += self.data[j][i]
            t += 1
        if t != 0 :
          partial_mean.append(float(s)/float(t))
        else:
          partial_mean.append(float('inf'))
      final_mean.append(partial_mean)
    return final_mean

  def generate_k_clusters(self,folder):
    pool = ThreadPool(cpu_count())
    result = pool.map(self.read_image, folder)
    pool.close()
    pool.join()
    # Fix comparisons to handle both regular Python objects and NumPy arrays
    self.cluster = [r[0] for r in result if r[0] is not None]
    self.data = [r[1] for r in result if r[1] is not None]
    self.end = [r[2] for r in result if r[2] is not None]
    
    # Apply dimensionality reduction if requested
    if self.dim_reduction and len(self.data) > 0:
      print(f"Applying {self.dim_reduction} dimensionality reduction...")
      
      # Choose the dimensionality reduction method
      if self.dim_reduction == 'pca':
        self.dim_reducer = PCA(n_components=min(self.dim_components, len(self.data[0])))
      elif self.dim_reduction == 'tsne':
        self.dim_reducer = TSNE(n_components=min(self.dim_components, len(self.data[0])), n_iter=1000, random_state=42)
      else:
        print(f"Unknown dimensionality reduction method: {self.dim_reduction}")
        return
      
      # Apply the dimensionality reduction
      try:
        reduced_data = self.dim_reducer.fit_transform(np.array(self.data))
        print(f"Reduced feature dimensions from {len(self.data[0])} to {reduced_data.shape[1]}")
        self.data = [list(rd) for rd in reduced_data]
      except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        print("Using original features instead.")

  def rearrange_clusters(self):
    isover = False
    while(not isover):
      isover = True
      m = self.generate_k_means()
      for x in range(len(self.cluster)):
        dist = []
        for a in range(self.k):
          dist.append( self.manhattan_distance(self.data[x],m[a]) )
        _mindist = dist.index(min(dist))
        if self.cluster[x] != _mindist :
          self.cluster[x] = _mindist
          isover = False

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="path to image folder")
ap.add_argument("-k", "--kmeans", type=int, default=5, help="how many groups")
ap.add_argument("-r", "--resample", type=int, default=128, help="size to resample the image by")
ap.add_argument("-s", "--size", default=False, action="store_true", help="use size to compare images")
ap.add_argument("-m", "--move", default=False, action="store_true", help="move instead of copy")

# Advanced feature extraction options
ap.add_argument("--feature", type=str, default="histogram", choices=["histogram", "vit", "swin", "efficientnetv2"],
                help="feature extraction method (histogram=original method, vit=Vision Transformer, swin=Swin Transformer, efficientnetv2=EfficientNetV2)")

# Dimensionality reduction options
ap.add_argument("--dim-reduction", type=str, default=None, choices=[None, "pca", "tsne"],
                help="dimensionality reduction method (None=no reduction, pca=Principal Component Analysis, tsne=t-SNE)")
ap.add_argument("--dim-components", type=int, default=50, 
                help="number of components to reduce to when using dimensionality reduction")
args = vars(ap.parse_args())
types = ('*.jpg', '*.JPG', '*.png', '*.jpeg')
imagePaths = []
folder = args["folder"]
if not folder.endswith("/"):
    folder+="/"
for files in types:
    imagePaths.extend(sorted(glob.glob(folder+files)))
nimages = len(imagePaths)
nfolders = int(math.log(args["kmeans"], 10))+1
if nimages <= 0:
    print("No images found!")
    exit()
if args["resample"] < 16 or args["resample"] > 256:
    print("-r should be a value between 16 and 256")
    exit()

# Feature extraction validation
if args["feature"] != "histogram" and not TIMM_AVAILABLE:
    print("Warning: Advanced feature extraction methods require the 'timm' library.")
    print("Falling back to histogram method.")
    args["feature"] = "histogram"

# Display feature extraction method
print(f"Using feature extraction method: {args['feature']}")
if args["dim_reduction"]:
    print(f"Applying dimensionality reduction: {args['dim_reduction']} with {args['dim_components']} components")

pbar = tqdm(total=nimages)
k = K_means(
    k=args["kmeans"],
    size=args["size"],
    resample=args["resample"],
    feature_method=args["feature"],
    dim_reduction=args["dim_reduction"],
    dim_components=args["dim_components"]
)
k.generate_k_clusters(imagePaths)
k.rearrange_clusters()
for i in range(k.k):
    try:
        os.makedirs(folder+str(i+1).zfill(nfolders))
    except:
        print("Folder already exists")
action = shutil.copy
if args["move"]:
    action = shutil.move
for i in range(len(k.cluster)):
    action(k.end[i], folder+"/"+str(k.cluster[i]+1).zfill(nfolders)+"/")

print("\nDone! Images organized into", k.k, "groups using", args["feature"], "features.")
