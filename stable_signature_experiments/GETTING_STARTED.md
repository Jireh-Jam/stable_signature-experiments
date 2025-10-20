# 🚀 Getting Started Guide

**Welcome to the Watermark Testing Pipeline!** This guide will help you run your first watermark robustness test in just a few minutes.

---

## ⚡ Quick Setup (5 minutes)

### 1. 📥 Download and Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/watermark-testing-pipeline
cd watermark-testing-pipeline

# Run the setup script
python setup.py
```

The setup script will:
- ✅ Check your Python version
- 📦 Install required packages
- 📁 Create necessary folders
- 🔽 Download watermark models (~200MB)
- ⚙️ Create a sample configuration

### 2. 📝 Update Your Settings

Edit the configuration file:
```bash
nano experiments/configs/user_config.yaml
```

**Change this line:**
```yaml
user:
  name: "Your.Username"  # Change to your actual username!
```

### 3. 🖼️ Add Test Images

Place 3-5 test images in the raw images folder:
```bash
# Copy your images here
cp /path/to/your/images/* experiments/data/raw/
```

**Supported formats:** `.jpg`, `.png`, `.bmp`, `.tiff`

### 4. 🚀 Run Your First Test

```bash
# Start Jupyter Notebook
jupyter notebook

# Open: pipeline_mk4_user_friendly.ipynb
# Follow the step-by-step instructions
```

---

## 📋 Step-by-Step Walkthrough

### Section 1: Configuration ⚙️
- Update your username
- Choose watermarking method (Stable Signature recommended)
- Set number of images to process (start with 5)

### Section 2: Install Packages 📦
- Loads required Python libraries
- Creates folder structure
- Usually runs automatically

### Section 3: Download Images 🌐
- **Skip this section** if you have your own images
- Only needed for Azure Blob Storage users

### Section 4: Load Models 🔧
- Downloads and loads watermarking models
- Takes 2-3 minutes first time
- Models are cached for future use

### Section 5: Add Watermarks 🔏
- Embeds invisible watermarks in your images
- Creates watermarked versions
- Shows progress for each image

### Section 6: Apply Transformations 🔄
- Tests 15+ different image modifications
- Crops, blurs, brightens, resizes images
- Creates folders for each transformation

### Section 7: Test Detection 🔍
- Checks if watermarks survive transformations
- Calculates confidence scores
- Shows detection results

### Section 8: Generate Report 📊
- Creates comprehensive analysis
- Generates charts and statistics
- Saves results to CSV files

### Section 9: Visualise Results 📈
- Creates professional charts
- Shows detection rates and confidence
- Saves plots as PNG files

### Section 10: Clean Up 🧹
- Optional: removes temporary files
- Organises final results

---

## 📊 Understanding Your Results

### 🎯 Key Metrics

**Detection Rate**: Percentage of successful watermark detections
- 🟢 **90%+**: Excellent robustness
- 🟡 **70-89%**: Good robustness  
- 🟠 **50-69%**: Moderate robustness
- 🔴 **<50%**: Poor robustness

**Confidence Score**: How certain the detector is (0.0 to 1.0)
- Higher scores = more confident detection
- Lower scores = less certain

### 📁 Output Files

After running the pipeline, check these folders:

```
experiments/
├── data/
│   ├── watermarked/          # Images with watermarks
│   └── transformed/          # Images after transformations
└── results/
    ├── *.csv                 # Detailed results data
    ├── *.png                 # Charts and graphs
    └── *.json                # Summary statistics
```

### 📈 Key Files to Check

1. **`watermark_detection_results.csv`** - Complete test results
2. **`detection_summary.csv`** - Summary by transformation
3. **`watermark_analysis_charts.png`** - Visual results
4. **`comparison_results.csv`** - Image quality metrics

---

## 🛠️ Troubleshooting

### ❌ Common Issues

**"Model file not found"**
- Run: `python setup.py` to download models
- Check: `ls models/checkpoints/` should show `.pt` files

**"No images found"**
- Add images to: `experiments/data/raw/`
- Supported formats: `.jpg`, `.png`, `.bmp`, `.tiff`

**"Permission denied"**
- Make sure you have write permissions
- Try: `chmod +x setup.py`

**"Out of memory"**
- Reduce `max_images_to_process` in config
- Use smaller images (resize to 512x512)

### 🔧 Configuration Issues

**Wrong username path**
- Update `user.name` in config file
- Make sure the path exists in your system

**Azure paths not working**
- Update `user.azure_root_dir` in config
- Use absolute paths if needed

### 📞 Getting Help

1. **Check the logs**: Look for error messages in notebook output
2. **Read the README**: Full documentation available
3. **Open an issue**: Report bugs on GitHub
4. **Ask the community**: Use GitHub Discussions

---

## 🎯 Next Steps

### 🔬 Run Advanced Tests

1. **Test more transformations**:
   ```yaml
   transformations:
     apply_aggressive: true  # More challenging tests
   ```

2. **Compare methods**:
   ```python
   # Test different watermarking methods
   for method in ["stable_signature", "trustmark", "watermark_anything"]:
       # Run pipeline with each method
   ```

3. **Custom transformations**:
   - Edit `tools/transformations.py`
   - Add your own image modifications

### 📊 Analyse Results

1. **Statistical analysis**: Use the CSV files with Excel/Python
2. **Custom plots**: Modify the visualisation code
3. **Comparative studies**: Test multiple methods side-by-side

### 🚀 Advanced Features

1. **Batch processing**: Process hundreds of images
2. **Cloud integration**: Run on Azure/AWS
3. **Custom models**: Train your own watermark detectors

---

## 💡 Tips for Best Results

### 🖼️ **Image Selection**
- Use diverse image types (photos, graphics, art)
- Include different sizes and aspect ratios
- Test both colour and grayscale images

### ⚙️ **Configuration**
- Start with small batches (5-10 images)
- Use standard transformations first
- Gradually increase complexity

### 📈 **Analysis**
- Focus on detection rates >70%
- Check confidence scores for reliability
- Look for patterns in vulnerable transformations

### 🔄 **Iteration**
- Run multiple tests with different settings
- Compare results across methods
- Document your findings

---

## 🎉 Congratulations!

You've successfully run your first watermark robustness test! 

**What you've accomplished:**
- ✅ Set up a professional watermarking pipeline
- ✅ Tested watermark robustness against 15+ transformations
- ✅ Generated comprehensive analysis and visualisations
- ✅ Created data for further research and development

**Ready for more?** Check out the [Advanced Usage Guide](docs/advanced_usage.md) or explore the [API Documentation](docs/api/).

---

<div align="center">

**🌟 Happy watermarking! 🌟**

[📚 Full Documentation](README.md) • [🤝 Contribute](CONTRIBUTING.md) • [💬 Get Help](https://github.com/your-repo/discussions)

</div>