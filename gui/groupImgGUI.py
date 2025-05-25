import os
import sys
import shutil
import glob
import math
import warnings
import numpy as np
import random
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Advanced feature extraction and dimensionality reduction imports
try:
    import torch
    import torchvision.transforms as transforms
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Import timm for advanced model support
    import timm
    TIMM_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    TORCH_AVAILABLE = False
    print("Warning: Advanced feature extraction requires torch, torchvision, and timm libraries.")
    print("Install with: pip install torch torchvision timm scikit-learn")

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore')

# Check for CUDA availability
if TORCH_AVAILABLE:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = None

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

  def __init__(self, k=3, size=False, resample=128, feature_method='histogram', dim_reduction=None, dim_components=50):
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
    self.model = get_feature_extractor(feature_method) if TORCH_AVAILABLE else None
    
    # Initialize transform for deep learning models
    if feature_method != 'histogram' and TORCH_AVAILABLE:
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
      if self.feature_method == 'histogram' or not TORCH_AVAILABLE:
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

class groupImgGUI(QWidget) :

	def __init__(self, parent = None) :		
		super(groupImgGUI, self).__init__(parent)	
		self.dir = None		
		self.progressValue = 0		
		self.createSettings()		
		layout = QVBoxLayout()
		self.btn = QPushButton("Select folder")
		self.btn.clicked.connect(self.selectFolder)			
		self.check = QCheckBox("Settings")
		self.check.stateChanged.connect(self.state);		
		self.runbtn = QPushButton("Run")
		self.runbtn.clicked.connect(self.run)	
		self.progress = QProgressBar(self)
		self.progress.hide()
		layout.addWidget(self.btn)
		layout.addWidget(self.check)		
		layout.addWidget(self.formGroupBox)	
		layout.addWidget(self.progress)	
		layout.addWidget(self.runbtn)	
		self.setMinimumSize(300,300)
		self.setLayout(layout)
		self.setWindowTitle("groupImg - GUI")

	def createSettings(self) :
		self.formGroupBox = QGroupBox("Settings")
		layout = QFormLayout()
		self.kmeans = QSpinBox()
		self.kmeans.setRange(3,15)
		self.kmeans.setValue(3)
		self.sample = QSpinBox()
		self.sample.setRange(32, 256)
		self.sample.setValue(128)
		self.sample.setSingleStep(2)
		self.move = QCheckBox()
		self.size = QCheckBox()
		self.visualization = QCheckBox()
		
		# Feature extraction method selection
		self.feature_method = QComboBox()
		self.feature_method.addItem("Histogram (Original)", "histogram")
		if TORCH_AVAILABLE and TIMM_AVAILABLE:
			self.feature_method.addItem("Vision Transformer (ViT)", "vit")
			self.feature_method.addItem("Swin Transformer", "swin")
			self.feature_method.addItem("EfficientNetV2", "efficientnetv2")
		else:
			self.feature_method.setToolTip("Advanced models require torch, torchvision and timm packages")
		
		# Dimensionality reduction method
		self.dim_reduction = QComboBox()
		self.dim_reduction.addItem("None", None)
		self.dim_reduction.addItem("PCA", "pca")
		self.dim_reduction.addItem("t-SNE", "tsne")
		
		# Components for dimensionality reduction
		self.dim_components = QSpinBox()
		self.dim_components.setRange(2, 100)
		self.dim_components.setValue(50)
		self.dim_components.setSingleStep(5)
		
		layout.addRow(QLabel("N. Groups:"), self.kmeans)
		layout.addRow(QLabel("Resample:"), self.sample)
		layout.addRow(QLabel("Move:"), self.move)
		layout.addRow(QLabel("Size:"), self.size)
		layout.addRow(QLabel("Visualization:"), self.visualization)
		layout.addRow(QLabel("Feature Method:"), self.feature_method)
		layout.addRow(QLabel("Dim. Reduction:"), self.dim_reduction)
		layout.addRow(QLabel("Dim. Components:"), self.dim_components)
		self.formGroupBox.hide()
		self.formGroupBox.setLayout(layout)
		
	def selectFolder(self) :
		QFileDialog.FileMode(QFileDialog.Directory)
		self.dir = QFileDialog.getExistingDirectory(self)
		self.btn.setText(self.dir or "Select folder")
		
	def state(self) :
		if self.check.isChecked() :
			self.formGroupBox.show()
		else:
			self.formGroupBox.hide()
	
	def disableButton(self) :
		self.runbtn.setText("Working...")
		self.runbtn.setEnabled(False)
	
	def enableButton(self) :		
		self.runbtn.setText("Run")
		self.runbtn.setEnabled(True)
		
	def visualize_clusters(self, k, image_paths):
		"""Display sample images from each cluster with clear visual separation"""
		# Determine number of clusters
		num_clusters = k.k
		
		# Get images grouped by cluster
		cluster_images = {}
		for i in range(len(k.cluster)):
			cluster_idx = k.cluster[i]
			image_path = k.end[i]
			if cluster_idx not in cluster_images:
				cluster_images[cluster_idx] = []
			cluster_images[cluster_idx].append(image_path)
		
		# Number of samples per cluster to display
		samples_per_cluster = 6
		num_cols = 3  # Number of columns for sample images
		
		# Calculate total number of valid clusters
		valid_clusters = []
		for cluster_idx in range(num_clusters):
			if cluster_idx in cluster_images and len(cluster_images[cluster_idx]) > 0:
				valid_clusters.append(cluster_idx)
		
		# Create figure with clear separation between clusters
		# Each cluster gets its own subplot with a grid of images inside
		# Increase height to provide more gap between rows
		fig = plt.figure(figsize=(15, 5 * len(valid_clusters)))
		fig.suptitle("Cluster Samples - Grouped by Similarity", fontsize=16, y=0.98)
		
		# Create subplots - one main subplot per cluster
		for idx, cluster_idx in enumerate(valid_clusters):
			# Get sample images for this cluster
			cluster_samples = cluster_images[cluster_idx]
			if len(cluster_samples) > samples_per_cluster:
				samples = random.sample(cluster_samples, samples_per_cluster)
			else:
				samples = cluster_samples
			
			# Create a subplot for this cluster
			cluster_plot = fig.add_subplot(len(valid_clusters), 1, idx + 1)
			cluster_plot.set_title(f"Cluster {cluster_idx+1} - {len(cluster_samples)} images", 
			                      fontsize=14, fontweight='bold', pad=20)
			cluster_plot.axis('off')
			
			# Add colored background to make clusters visually distinct
			cluster_colors = ['#f0f9e8', '#e6f5d0', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe']
			color_idx = cluster_idx % len(cluster_colors)
			cluster_plot.patch.set_facecolor(cluster_colors[color_idx])
			cluster_plot.patch.set_alpha(0.3)  # Make background semi-transparent
			
			# Create a grid of images within this cluster subplot
			for i, sample_path in enumerate(samples):
				if i >= samples_per_cluster:
					break
				
				# Create a nested subplot for each image
				ax = plt.subplot2grid((len(valid_clusters), num_cols), 
				                   (idx, i % num_cols), 
				                   fig=fig)
				
				try:
					# Load and display image
					img = Image.open(sample_path).convert('RGB')
					ax.imshow(img)
					
					# Remove filenames to keep visualization clean
					# No title for individual images
					ax.axis('off')
				except Exception as e:
					print(f"Error loading image {sample_path}: {e}")
		
		# Add feature extraction method and total images info
		feature_method = 'histogram'
		if hasattr(k, 'feature_method'):
			feature_method = k.feature_method
		
		total_images = sum(len(imgs) for imgs in cluster_images.values())
		fig.text(0.5, 0.01, 
		         f"Feature extraction: {feature_method} | Total images: {total_images}", 
		         ha='center', fontsize=10)
		
		# Adjust layout with much more space between clusters
		plt.subplots_adjust(hspace=1.0, wspace=0.3)
		plt.tight_layout(rect=[0, 0.02, 1, 0.96])
		plt.show()
			
	def run(self) :
		self.disableButton()
		types = ('*.jpg', '*.JPG', '*.png', '*.jpeg')
		imagePaths = []
		folder = self.dir
		if not folder.endswith("/") :
			folder+="/"
		for files in types :
			imagePaths.extend(sorted(glob.glob(folder+files)))
		nimages = len(imagePaths)
		nfolders = int(math.log(self.kmeans.value(), 10))+1		
		if nimages <= 0 :
			QMessageBox.warning(self, "Error", 'No images found!')
			self.enableButton()
			return
		
		# Get feature extraction and dimensionality reduction settings
		feature_method = self.feature_method.currentData()
		dim_reduction = self.dim_reduction.currentData()
		dim_components = self.dim_components.value()
		
		# Check if advanced models are available
		if feature_method != "histogram" and not (TORCH_AVAILABLE and TIMM_AVAILABLE):
			QMessageBox.warning(self, "Warning", 'Advanced feature extraction requires torch, torchvision and timm packages. Using histogram method instead.')
			feature_method = "histogram"
		
		# Initialize K-means with advanced options
		k = K_means(
			k=self.kmeans.value(),
			size=self.size.isChecked(),
			resample=self.sample.value(),
			feature_method=feature_method,
			dim_reduction=dim_reduction,
			dim_components=dim_components
		)
		
		# Show progress message
		if feature_method != "histogram":
			QMessageBox.information(self, "Processing", f'Using {feature_method} feature extraction with {dim_reduction if dim_reduction else "no"} dimensionality reduction.\nThis may take some time for large image sets.')
		
		k.generate_k_clusters(imagePaths)
		k.rearrange_clusters()
		for i in range(k.k) :
			try :
				os.makedirs(folder+str(i+1).zfill(nfolders))
			except Exception as e :
				print("Folder already exists", e)
		action = shutil.copy
		if self.move.isChecked() :
			action = shutil.move
		for i in range(len(k.cluster)):
			action(k.end[i], folder+"/"+str(k.cluster[i]+1).zfill(nfolders)+"/")
		# Show visualization if requested
		if self.visualization.isChecked():
			self.visualize_clusters(k, imagePaths)
		
		QMessageBox.information(self, "Done", 'Done!')
		self.enableButton()
	
def main():
   app = QApplication(sys.argv)
   groupimg = groupImgGUI()
   groupimg.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
