# AIgroupImg

AI-powered image organization tool using advanced feature extraction and machine learning for similarity-based clustering.


## Key Features

- **Neural Network Encoders** - Extract rich feature representations from images
- **Dimensionality Reduction** - PCA and t-SNE for better clustering
- **Visual Cluster Analysis** - View sample images from each cluster

## Use Cases

- **E-commerceThumbnail Feature Extraction** - Extract structured features from thumbnails for predictive modeling


## How to use

Install dependencies for CLI or GUI:

```bash
pip install -r cli/requirements.txt  # For CLI
pip install -r gui/requirements.txt  # For GUI with visualization
```

### CLI

Call the script passing the image folder you want to organize.

```bash
python cli/groupimg.py -f /path/to/images [-k NUM_CLUSTERS] [--feature MODEL] [options]
```

**Basic Options:**

- `-f PATH` - Folder containing images (required)
- `-k NUM` - Number of clusters (default: 3)
- `-m` - Move images instead of copying
- `-s` - Consider image size as a feature

**Advanced Options:**

- `--feature MODEL` - Feature extraction method (histogram, vit, swin, efficientnetv2)
- `--dim-reduction METHOD` - Dimensionality reduction (pca, tsne)
- `--dim-components NUM` - Number of components for dimension reduction (default: 50)

### GUI

![groupImgGUI](./demo/screenshot-GUI.png)

```bash
python gui/groupImgGUI.py
```

Select a folder of images and adjust settings as needed.

**Settings Options:**

- **N. Group** - Number of clusters
- **Feature Method** - Feature extraction method (histogram, ViT, Swin, EfficientNetV2)
- **Dim. Reduction** - Dimensionality reduction method (PCA, t-SNE)
- **Visualization** - Display sample images from each cluster

### Visualization

![Cluster Visualization](./demo/cluster.png)

When the visualization option is enabled, the application will display sample images from each cluster after processing is complete. This helps you quickly assess the quality of the clustering and how well the selected feature extraction method works for your image set.

## How It Works

1. **Feature Extraction** - Extract rich representations using:
   - **Histogram [original]** - Color distribution-based features
   - **Deep Learning Models** - ViT, Swin Transformer, EfficientNetV2
2. **Dimensionality Reduction** - Optionally apply PCA or t-SNE to the features
3. **Clustering** - Use K-means to group similar images
4. **Organization** - Sort images into folders by cluster

## Credits

- Original project by [Victor Ribeiro](https://github.com/victorqribeiro/groupImg)
- Enhanced with advanced feature extraction, dimensionality reduction, and visualization techniques
