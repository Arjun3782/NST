# 🎨 Neural Style Transfer with PyTorch

This project implements **Neural Style Transfer (NST)** — a deep learning technique that blends the content of one image with the style of another to generate a visually artistic result. Built using **PyTorch** and **VGG19**, it demonstrates how optimization can be applied directly to an image to reimagine it in a chosen artistic style.

---

## 📌Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Equations Used](#equations-used)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)

---

## 📖 Overview

- Extracts **content** features from a base image.
- Captures **style** features from a reference painting.
- Combines both using optimization to create a new image that looks like the content painted in the style.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=YOUR_IMAGE_ID" width="600"/>
</p>

---

## 🧠 How It Works

1. Use a **pre-trained VGG19** network to extract features.
2. Compute:
   - **Content Loss**: difference between content and generated image features.
   - **Style Loss**: difference between Gram matrices of style and generated images.
3. Combine both losses to compute **total loss**.
4. Use the **Adam optimizer** to update the pixels of the generated image.

---

## ✍️ Equations Used

 ![Screenshot 2025-05-24 111541](https://github.com/user-attachments/assets/440772e2-20e4-449a-b2a8-c20a5c955415)
- **Content Loss**:  
  `L_content = (1/2) × ∑(F − P)²`

- **Gram Matrix**:  
  `G = (F × Fᵀ) / (C × H × W)`

- **Style Loss**:  
  `L_style = ∑ₗ wₗ × ‖Gₗ^g − Gₗ^a‖²`

- **Total Loss**:  
  `L_total = α × L_content + β × L_style`

- **Adam Optimizer Update Rule**:  
  `θₜ₊₁ = θₜ − η × (mₜ / (√vₜ + ε))`

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Arjun3782/NST.git
cd NST

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

##  Results 
![Screenshot 2025-05-26 083705](https://github.com/user-attachments/assets/4f8e93bf-7226-4f3f-a718-c6d5d8a2df58)

![Screenshot 2025-05-26 082204](https://github.com/user-attachments/assets/a453280f-2b81-4fa7-8091-4717d83098fe)
