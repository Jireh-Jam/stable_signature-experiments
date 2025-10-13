# Quick Start Guide

Get started with the Watermark Robustness Testing Pipeline in just a few minutes!

## ⚡ For the Impatient

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the notebook
jupyter notebook pipeline_mk4.ipynb

# 3. In the notebook, update Section 1:
#    - Set WATERMARK_METHOD to your chosen method
#    - Update USER_NAME if using Azure AI
#    - Set MAX_IMAGES_TO_PROCESS = 10 for testing

# 4. Run all cells sequentially

# 5. Check results in outputs/{method}/results/
```

## 📝 Step-by-Step Guide

### Step 1: Choose Your Watermarking Method

Decide which watermarking system you want to test:

- **TrustMark** - Easiest to get started (no model downloads required)
- **Watermark Anything** - Requires model checkpoint
- **Stable Signature** - Requires model download from Meta

### Step 2: Set Up Your Chosen Method

#### For TrustMark (Recommended for First-Time Users)

```bash
# Install TrustMark
pip install trustmark

# Create watermarked images directory
mkdir -p watermark_models/trustmark/watermarked_images

# Generate a test watermarked image
python3 << EOF
from trustmark import TrustMark
from PIL import Image
import requests
from io import BytesIO

# Initialise TrustMark
tm = TrustMark(verbose=False, model_type='Q')

# Download a sample image
url = "https://picsum.photos/512/512"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert('RGB')

# Create watermark and encode
watermark_bits = [1, 0, 1, 1, 0, 1, 0, 0] * 4  # 32 bits
watermarked = tm.encode(image, watermark_bits)

# Save
watermarked.save('watermark_models/trustmark/watermarked_images/test_001.png')
print("✓ Test image created!")
EOF
```

#### For Stable Signature

```bash
# Download the pre-trained decoder
mkdir -p watermark_models/stable-signature/models
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt \
     -P watermark_models/stable-signature/models/

# You'll need to generate watermarked images using the fine-tuned decoder
# See watermark_models/stable-signature/README.md for details
```

#### For Watermark Anything

```bash
# Clone the Watermark Anything repository (if not already done)
# Follow their setup instructions
# Place your model checkpoint in watermark_models/watermark-anything/models/
```

### Step 3: Configure the Notebook

Open `pipeline_mk4.ipynb` and update Section 1:

```python
# Choose your method
WATERMARK_METHOD = "TrustMark"  # or "Stable_Signature" or "Watermark_Anything"

# Update your username (if using Azure AI)
USER_NAME = 'YourName'  # Change this!

# Start with a small number for testing
MAX_IMAGES_TO_PROCESS = 10
```

### Step 4: Run the Pipeline

1. **Execute cells in order** - Don't skip sections
2. **Read the instructions** in each section
3. **Monitor progress bars** to track processing
4. **Check for errors** - the notebook will alert you if something goes wrong

### Step 5: Review Results

Your results will be saved in:
```
outputs/{method}/
├── transformed_images/     # All transformed images organized by type
│   ├── resize/
│   ├── crop/
│   ├── rotation/
│   ├── blur/
│   └── ...
└── results/
    ├── watermark_detection_results_{method}.csv
    └── results_visualisation_{method}.png
```

Open the CSV file in Excel, Google Sheets, or any spreadsheet application to analyse:
- Which transformations break watermark detection
- How image quality metrics correlate with detection
- Overall robustness of your watermarking method

## 🎯 What to Expect

### Processing Time

For 10 images with all transformations enabled:
- **TrustMark**: ~2-5 minutes (CPU) or ~1-2 minutes (GPU)
- **Watermark Anything**: ~3-6 minutes (CPU) or ~1-3 minutes (GPU)
- **Stable Signature**: ~3-6 minutes (CPU) or ~1-3 minutes (GPU)

Actual times depend on:
- Image resolution
- Number of transformations enabled
- Your hardware specifications

### Disk Space Required

- **Models**: 100MB - 2GB (depending on method)
- **Per image processed**: ~20MB (includes all transformations)
- **Results and metrics**: <10MB per run

For 10 images: Plan for ~300MB total
For 100 images: Plan for ~2.5GB total

## ✅ Success Checklist

After running the pipeline, verify:

- [ ] CSV results file created in `outputs/{method}/results/`
- [ ] Transformed images saved in subdirectories
- [ ] Visualisation PNG generated
- [ ] No error messages in notebook output
- [ ] Detection rate summary printed
- [ ] Metrics calculated for all images

## 🔍 Quick Verification

Run this to verify your setup before processing all images:

```python
from config import create_config

config = create_config(
    method="TrustMark",  # Your chosen method
    user_name="YourName",
    max_images=5
)

config.print_configuration()
```

Look for "✓ Configuration is valid" in the output.

## 💡 Pro Tips

1. **Test with 5-10 images first** to ensure everything works
2. **Enable only a few transformations** initially to speed up testing
3. **Check the CSV after each run** to ensure detection is working
4. **Save your configuration** if you find settings that work well
5. **Use a GPU if available** for 2-3x faster processing

## 🆘 Quick Troubleshooting

**Problem**: "No images found"
- **Solution**: Make sure images are in `watermark_models/{method}/watermarked_images/`

**Problem**: "Model checkpoint not found"
- **Solution**: Download the model following the method-specific README

**Problem**: Slow processing
- **Solution**: Reduce `MAX_IMAGES_TO_PROCESS` or disable some transformations

**Problem**: Out of memory
- **Solution**: Process fewer images at once or use a smaller image size

## 📚 Next Steps

Once you've successfully run the pipeline:

1. **Experiment with transformations** - Try different parameters
2. **Compare methods** - Run the pipeline with each watermarking method
3. **Analyse results** - Use the CSV data for detailed analysis
4. **Add your own method** - See `watermark_models/README.md`

## 🎓 Learning Resources

- **Notebook Documentation**: Each cell includes detailed British English instructions
- **Method READMEs**: Technical details in `watermark_models/{method}/README.md`
- **Main README**: Comprehensive guide in `README.md`
- **Configuration Guide**: See `config.py` for all options

---

Need more help? Check the main [README.md](README.md) or open an issue on GitHub.

**Happy testing! 🚀**
