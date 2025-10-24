# Rasterized Steered Mixture of Experts for Efficient 2D Image Regression

Official implementation of the paper
**“Rasterized Steered Mixture of Experts for Efficient 2D Image Regression.”**

This repository introduces a compact and efficient image regression method that represents a scene using a set of **Gaussian density functions**. It also includes a **multi-model fusion approach** for image denoising.

---

## 📂 Data Preparation

1. Rename your target image to `0000001.png`.
2. Place it in the folder:

   ```
   <source_image_path>/images/
   ```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yihsinli/r-smoe.git --recursive

# Use your existing 3DGS environment, or create a new one
conda env create --file my_environment.yml
conda activate r-smoe
```

---

## 🧠 Training and Testing a Single Image

To train and test a single image, run:

```bash
python train_render_metrics_final.py -s <source_image_path> -m <model_path> \
  --npcs <num_kernels> --iterations <num_iterations> --file_name <file_name>
```

**Arguments:**

* `-s` : Source image path
* `-m` : Model output path
* `--npcs` : Number of kernels
* `--iterations` : Number of training iterations
* `--file_name` : Output filename

---

## 🧩 Denoising with Multi-Model Fusion

To perform denoising using multi-model fusion, run:

```bash
python train_noisy_final.py -s <source_image_path> -m <model_path> \
  --npcs <num_kernels> --iterations <num_iterations> --file_name <file_name> \
  --n_multi_model <num_models>
```

**Additional Argument:**

* `--n_multi_model` : Number of models used for fusion

---

## 💡 Parameter Recommendations

* For denoising, use fewer kernels to prevent over-reconstruction of noise:
  `--npcs=1000`
* Reduce training iterations (e.g., `--iterations=300`) for noisy inputs.
* Increasing `--n_multi_model` improves denoising quality but increases runtime.
  **Recommended:** `--n_multi_model=8` for a balance between accuracy and efficiency.

**Outputs:**

* Reconstructed image:
  ```
  <model_path>/train/ours_<iterations>/renders/00000.png
  ```

* Denoised image:

  ```
  <model_path>/train/ours_<iterations>/00001.png
  ```
* Metrics and runtime summary:

  ```
  <all_results>/<file_name>.json
  ```

---

## 🙏 Acknowledgements

This project builds upon [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting).
We thank the authors for providing such a strong foundation.

---

## 📖 Citation

If you find our work helpful, please cite:

```bibtex
@misc{li2025rasterizedsteeredmixtureexperts,
      title={Rasterized Steered Mixture of Experts for Efficient 2D Image Regression},
      author={Yi-Hsin Li and Thomas Sikora and Sebastian Knorr and Mårten Sjöström},
      year={2025},
      eprint={2510.05814},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.05814}
}
```