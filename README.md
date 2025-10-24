# Rasterized Steered Mixture of Experts for Efficient 2D Image Regression

This repository provides the official implementation of the paper **“Rasterized Steered Mixture of Experts for Efficient 2D Image Regression.”**
The method represents a scene using a set of Gaussian density functions to improve compactness and efficiency for image regression.
In addition, we provide multi-model fusion methods for image denoising.

---

## 📂 Data Preparation

1. Rename your target image to `0000001.png`.

2. Place it in the following directory structure:

   ```
   data/<image_name>/images/
   ```

3. If you want to test multiple images, simply duplicate the folder `<image_name>` as shown below:

   ```
   data/
   ├── image_name1/
   │   ├── images/
   │   └── sparse/
   ├── image_name2/
   │   ├── images/
   │   └── sparse/
   └── ...
   ```

4. When running the code, specify the folder path using:

   ```
   -s data/<image_name>
   ```

   For example:

   ```
   -s data/image_name1
   ```

---

## ⚙️ Installation

```bash
# Download repository
git clone https://github.com/yihsinli/r-smoe.git --recursive

# If you already have an environment used for 3DGS, you can use it
# Otherwise, create a new environment
conda env create --file my_environment.yml
conda activate r-smoe
```

---

## 🚀 Training and Testing on a Single Image

To train and test an image:

```bash
python train_render_metrics_final.py -s <source_image_path> -m <model_path> --npcs <k> --iterations <iter> --file_name <file_name>
```

**Command-line arguments:**

```
-s              # Source image path (e.g., data/image_name1)
-m              # Model path
--npcs          # Number of kernels
--iterations    # Number of training iterations
--file_name     # Result file name
```

---

## 🧩 Denoising with Multi-Model Fusion

To train and test the denoising method:

```bash
python train_noisy_final.py -s <source_image_path> -m <model_path> --npcs <k> --iterations <iter> --file_name <file_name> --n_multi_model <n_mm>
```

**Command-line arguments:**

```
-s              # Source image path (e.g., data/image_name1)
-m              # Model path
--npcs          # Number of kernels
--iterations    # Number of training iterations
--file_name     # Result file name
--n_multi_model # Number of models for fusion
```

---

## 💡 Tips for Parameter Adjustment

* For **denoising**, use a smaller number of kernels to prevent overfitting noise, e.g.:

  ```
  --npcs=1000
  ```
* Reduce training iterations (e.g. to 300) to avoid overreconstruction.
* More models (`--n_multi_model`) improve denoising performance but increase runtime.
  We recommend using **8 models** for a good accuracy–efficiency balance.

**Output files:**
* Reconstructed image:
  `<model_path>/train/ours_<iterations>/renders/00000.png`
* Denoised image:
  `<model_path>/train/ours_<iter>/00001.png`
* Quantitative results (metrics, training time, reconstruction time):
  `<all_results>/<file_name>.json`

---

## 🙏 Acknowledgements

This project is built upon [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting).
We thank the original authors for their excellent work.

---

## 📖 Citation

If you find our code or paper useful, please consider citing:

```bibtex
@misc{li2025rasterizedsteeredmixtureexperts,
      title={Rasterized Steered Mixture of Experts for Efficient 2D Image Regression}, 
      author={Yi-Hsin Li and Thomas Sikora and Sebastian Knorr and Mårten Sjöström},
      year={2025},
      eprint={2510.05814},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.05814}, 
}
```
