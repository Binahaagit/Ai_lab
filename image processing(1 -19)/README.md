# 🖼️ Digital Image Processing — Lab Experiments

### 🐍 Python + OpenCV + NumPy + Matplotlib
### 📚 Experiments 1–19 | Quick Revision & Exam Guide

---

## ⚙️ General Instructions

- Keep the **Python `.py` file and input image in the same folder**.
- If the image is elsewhere, use the full path:

```python
img = cv2.imread(r"C:\Users\Downloads\monkey.jpg")
```

- Always check the image filename and extension.
- For Windows paths, use `r"..."` before the path.
- In Python IDLE: `File → Open → Run → Run Module` or press **F5**.

### 📦 Required Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```

| Library | Use |
|---|---|
| `cv2` | Image processing |
| `numpy` | Arrays and calculations |
| `matplotlib.pyplot` | Histograms, plots, subplots |
| `os` | File-size operations |

### 🖥️ Display

```python
cv2.imshow()
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```python
plt.imshow()
plt.subplot()
plt.show()
```

- `cv2.imshow()` → display an image
- `cv2.waitKey(0)` → wait for a key press
- `cv2.destroyAllWindows()` → close OpenCV windows
- `plt.subplot()` → arrange multiple outputs
- `plt.show()` → display Matplotlib output
- `plt.tight_layout()` → adjust subplot spacing

### 🚨 If the Image Is Not Loading

```python
print(img is None)
```

- `True` → image was not read
- `False` → image was read successfully

Check the **path, filename and extension**.

---

# 🧠 Basic Things to Remember

### 🖼️ OpenCV Colour Order

```text
OpenCV      → BGR
Matplotlib  → RGB
```

For Matplotlib:

```python
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### 🔢 8-bit Images

```text
0   → Black
255 → White
```

Normal pixel range: `0–255`

### 📐 Common Kernel Sizes

```text
3 × 3
5 × 5
7 × 7
```

```text
3×3 → 9 pixels
5×5 → 25 pixels
7×7 → 49 pixels
```

---

# 1️⃣ Experiment 1 — Read, Display, Save, Grayscale & Binary

### 🔑 Important Functions

```python
cv2.imread()
cv2.imshow()
cv2.imwrite()
cv2.cvtColor()
cv2.threshold()
cv2.waitKey()
cv2.destroyAllWindows()
```

- `imread()` → read image
- `imshow()` → display image
- `imwrite()` → save image
- `cvtColor()` → colour → grayscale
- `threshold()` → grayscale → binary
- Binary → `0` or `255`

---

# 2️⃣ Experiment 2 — Grayscale & Binary Without Conversion Functions

### 🚨 Important

Do **not** use OpenCV conversion functions for the actual grayscale/binary computation.

### 🎨 Grayscale Formula

```text
Gray = 0.114B + 0.587G + 0.299R
```

### ⚫ Binary Conversion

```text
Gray > 127   → 255
Gray ≤ 127  → 0
```

### 🔑 Remember

```python
np.zeros()
```

- `img.shape[0]` → height
- `img.shape[1]` → width
- `B, G, R = img[i, j]`

---

# 3️⃣ Experiment 3 — Image Transformations

### 🔑 Important Functions

```python
cv2.warpAffine()
cv2.resize()
cv2.getRotationMatrix2D()
```

- Translation → move image
- Scaling → resize image
- Rotation → rotate image

Translation matrix:

```text
[[1, 0, tx],
 [0, 1, ty]]
```

- `tx` → horizontal movement
- `ty` → vertical movement

---

# 4️⃣ Experiment 4 — Image Addition, Subtraction & Blending

### 🔑 Important Functions

```python
cv2.add()
cv2.subtract()
cv2.addWeighted()
cv2.resize()
```

- `add()` → image addition
- `subtract()` → image subtraction
- `addWeighted()` → blending

```python
cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
```

→ 50% image 1 + 50% image 2

📌 Images should have the **same size and channels**.

---

# 5️⃣ Experiment 5 — Binary AND, OR, XOR & Shapes

### 🔑 Important Functions

```python
cv2.rectangle()
cv2.circle()
cv2.bitwise_and()
cv2.bitwise_or()
cv2.bitwise_xor()
```

```text
AND → White only when BOTH are white
OR  → White when AT LEAST ONE is white
XOR → White when they are DIFFERENT
```

```text
0   → Black
255 → White
```

---

# 6️⃣ Experiment 6 — 4-Neighbours & 8-Neighbours

### 🔲 4-Neighbours

```text
       TOP
        ↑
LEFT ← PIXEL → RIGHT
        ↓
      BOTTOM
```

### 🔳 8-Neighbours

4-neighbours + 4 diagonal neighbours.

### 📝 Remember

- Count neighbours having the **same colour** as the current pixel.
- `0` → black
- `255` → white
- Check image boundaries.
- Neighbour calculation must be done **manually**.
- No library routine should be used for the actual neighbour computation.

### Directions

```python
d4 = [(-1,0), (1,0), (0,-1), (0,1)]

d8 = [(-1,0), (1,0), (0,-1), (0,1),
      (-1,-1), (-1,1), (1,-1), (1,1)]
```

---

# 7️⃣ Experiment 7 — Euclidean & Manhattan Distance

### 📏 Euclidean Distance

```text
√((x₂-x₁)² + (y₂-y₁)²)
```

### 🏙️ Manhattan Distance

```text
|x₂-x₁| + |y₂-y₁|
```

- Euclidean → straight-line distance
- Manhattan → grid/path distance
- No library routine required.

---

# 8️⃣ Experiment 8 — Image Negative

### 🔑 Formula

```text
s = 255 - r
```

- Binary → `0 ↔ 255`
- Grayscale → invert every pixel
- Colour → invert B, G and R separately

```python
negative = 255 - image
```

---

# 9️⃣ Experiment 9 — Log Transformation

### 🔑 Formula

```text
s = c × log(1 + r)
```

- `np.log()` → logarithm
- `+1` → avoids `log(0)`
- Enhances details in **dark regions**
- Compresses bright intensity values

```python
c = 255 / np.log(1 + 255)
```

---

# 🔟 Experiment 10 — Gamma Correction

### 🔑 Formula

```text
s = r^γ
```

### 🔄 Process

```text
Pixel 0–255
     ↓
Normalize /255
     ↓
Power γ
     ↓
Multiply ×255
     ↓
Convert to uint8
```

```text
γ < 1 → Brighter
γ = 1 → Unchanged
γ > 1 → Darker
```

### 🎯 Required Values

```text
0.2, 0.5, 1.5, 2, 3
```

### 🔑 Function

```python
np.power()
```

---

# 1️⃣1️⃣ Experiment 11 — Histogram Equalization on Grayscale Image

### 🔑 Function

```python
cv2.equalizeHist()
```

- Used for grayscale images
- Improves image contrast
- Histogram shows the frequency of intensity values
- `ravel()` → converts image matrix to 1D
- `256` → intensity levels `0–255`

### 🧠 Manual Method

```text
Histogram
    ↓
CDF
    ↓
Find CDFmin
    ↓
Calculate new intensity
    ↓
Replace pixels
```

🚨 If the question specifically asks for a **manual** method, do not use `cv2.equalizeHist()` for the actual computation.

---

# 1️⃣2️⃣ Experiment 12 — Histogram Equalization on Colour Image

### 🔄 Main Process

```text
BGR
 ↓
HSV
 ↓
Separate H, S, V
 ↓
Equalize V
 ↓
Merge H, S, V
 ↓
HSV → BGR
```

```text
H → Hue → Colour
S → Saturation → Colour intensity
V → Value → Brightness
```

### 🚨 Important

**Equalize only the V channel** so that the colours are not unnecessarily distorted.

### 🔑 Functions

```python
cv2.cvtColor()
cv2.split()
cv2.equalizeHist()
cv2.merge()
```

For Matplotlib display:

```python
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

---

# 🗜️ COMPRESSION EXPERIMENTS

# 1️⃣3️⃣ Experiment 13 — JPEG Compression

### 🎯 Quality Levels

```text
90, 70, 50, 30, 10
```

### 🔑 Flag

```python
cv2.IMWRITE_JPEG_QUALITY
```

### 🔑 Functions

```python
cv2.imwrite()
os.path.getsize()
```

```text
JPEG → Lossy
Quality ↓ → File size generally ↓
Quality ↓ → Image quality generally ↓
```

```python
size = os.path.getsize(filename)
size_kb = size / 1024
```

- `getsize()` → size in bytes
- `/1024` → converts bytes to KB

```python
cv2.imwrite(filename, img,
            [cv2.IMWRITE_JPEG_QUALITY, q])
```

---

# 1️⃣4️⃣ Experiment 14 — PNG Compression

### 🎯 Compression Levels

```text
0, 3, 5, 7, 9
```

### 🔑 Flag

```python
cv2.IMWRITE_PNG_COMPRESSION
```

```text
PNG → Lossless
Level 0 → No compression
Level 9 → Maximum compression
```

```text
JPEG → Quality
PNG  → Compression level
```

Compression ↑ → File size generally ↓

Image quality → Remains unchanged

---

# 1️⃣5️⃣ Experiment 15 — WEBP Compression

### 🎯 Quality Levels

```text
100, 70, 50, 30, 10
```

### 🔑 Flag

```python
cv2.IMWRITE_WEBP_QUALITY
```

- WEBP → efficient image format
- Supports lossy and lossless compression
- Quality ↓ → generally file size ↓
- Quality ↓ → generally image quality ↓ when using lossy quality mode

```python
cv2.imwrite(filename, img,
            [cv2.IMWRITE_WEBP_QUALITY, q])
```

---

# 🌫️ FILTERING EXPERIMENTS

# 1️⃣6️⃣ Experiment 16 — Spatial Averaging Filter

### 🔑 Function

```python
cv2.filter2D()
```

### 🧮 Kernels

```text
3×3 → divide by 9
5×5 → divide by 25
7×7 → divide by 49
```

```text
Spatial averaging → Smooth / Blur
```

All neighbouring pixels have **equal weights**.

```python
kernel3 = np.ones((3,3), np.float32) / 9
```

Larger kernel → stronger smoothing.

---

# 1️⃣7️⃣ Experiment 17 — Gaussian Blur

### 🔑 Function

```python
cv2.GaussianBlur()
```

### 🎯 Kernel Sizes

```text
3×3
5×5
7×7
```

```python
cv2.GaussianBlur(img, (3,3), 0)
```

- Smooths image
- Reduces noise
- Uses **Gaussian-weighted values**
- Nearby pixels have greater influence
- Larger kernel → stronger smoothing
- `sigmaX = 0` → automatically calculated

```text
Averaging → Equal weights
Gaussian  → Weighted values
```

---

# 1️⃣8️⃣ Experiment 18 — Median Blurring

### 🔑 Function

```python
cv2.medianBlur()
```

### 🎯 Kernel Sizes

```text
3×3 → 3
5×5 → 5
7×7 → 7
```

- Smooths image
- Good for removing noise
- Replaces a pixel with the **median value** of its neighbourhood
- Larger kernel → stronger smoothing

```python
cv2.medianBlur(img, 3)
```

📌 Unlike `GaussianBlur()`, `medianBlur()` takes **one number** for the kernel size.

---

# 1️⃣9️⃣ Experiment 19 — Laplacian Filter

### 🔑 Functions

```python
cv2.Laplacian()
cv2.convertScaleAbs()
```

### 🎯 Kernel Sizes

```text
3×3
5×5
7×7
```

- Laplacian → **EDGE DETECTION**
- Input should be **grayscale**
- `CV_64F` → safely handles positive and negative values
- `convertScaleAbs()` → converts result to 8-bit for display

### 🔄 Process

```text
Original image
      ↓
Laplacian calculation
      ↓
CV_64F → safely handles negative values
      ↓
convertScaleAbs()
      ↓
8-bit image (0–255)
      ↓
Display
```

### Why `CV_64F`?

Laplacian can produce both negative and positive values. Normal `uint8` only stores `0–255`, so `CV_64F` safely stores the calculation.

### Why `convertScaleAbs()`?

It converts the result into a suitable **8-bit image (0–255)** for display.

---

# ⚡ LAST-MINUTE FUNCTION CHEAT SHEET

| Exp. | Topic | ⭐ Remember |
|---|---|---|
| 1 | Read / Gray / Binary | `imread()`, `cvtColor()`, `threshold()` |
| 2 | Manual Gray / Binary | `np.zeros()` + loops |
| 3 | Transformations | `warpAffine()`, `resize()`, `getRotationMatrix2D()` |
| 4 | Add / Subtract / Blend | `add()`, `subtract()`, `addWeighted()` |
| 5 | AND / OR / XOR | `bitwise_and/or/xor()` |
| 6 | Neighbours | Manual calculation |
| 7 | Distances | Euclidean + Manhattan formulas |
| 8 | Negative | `255 - pixel` |
| 9 | Log | `np.log()` |
| 10 | Gamma | `np.power()` |
| 11 | Gray Histogram | `equalizeHist()` |
| 12 | Colour Histogram | HSV + Equalize V |
| 13 | JPEG | `IMWRITE_JPEG_QUALITY` |
| 14 | PNG | `IMWRITE_PNG_COMPRESSION` |
| 15 | WEBP | `IMWRITE_WEBP_QUALITY` |
| 16 | Averaging | `filter2D()` |
| 17 | Gaussian | `GaussianBlur()` |
| 18 | Median | `medianBlur()` |
| 19 | Laplacian | `Laplacian()` |

---

# 🚨 30-SECOND MEMORY MAP

```text
1️⃣  READ / GRAY / BINARY
2️⃣  MANUAL GRAY / BINARY
3️⃣  TRANSFORM
4️⃣  ADD / SUBTRACT / BLEND
5️⃣  AND / OR / XOR
6️⃣  NEIGHBOURS
7️⃣  DISTANCES
8️⃣  NEGATIVE
9️⃣  LOG
🔟 GAMMA
1️⃣1️⃣ HISTOGRAM — GRAY
1️⃣2️⃣ HISTOGRAM — COLOUR / HSV
1️⃣3️⃣ JPEG
1️⃣4️⃣ PNG
1️⃣5️⃣ WEBP
1️⃣6️⃣ AVERAGING
1️⃣7️⃣ GAUSSIAN
1️⃣8️⃣ MEDIAN
1️⃣9️⃣ LAPLACIAN
```

---

# 🧠 MOST IMPORTANT DIFFERENCES

```text
Grayscale      → One intensity channel
Binary         → 0 or 255

JPEG           → Lossy + Quality
PNG            → Lossless + Compression level
WEBP           → Quality / efficient compression

Averaging      → Equal weights
Gaussian       → Gaussian weights
Median         → Median value

Laplacian      → Edge detection

HSV:
H → Hue
S → Saturation
V → Brightness
```

---

# 🚨 EXAM RULES TO NEVER FORGET

- 📁 Check the image path **before running**.
- 🖼️ If `imread()` returns `None`, fix the path.
- 🔢 Normal 8-bit image → pixel values `0–255`.
- 🪟 `cv2.imshow()` → display image.
- ⏳ `cv2.waitKey(0)` → wait for key.
- ❌ `cv2.destroyAllWindows()` → close OpenCV windows.
- 📊 `plt.show()` → display Matplotlib output.
- 📐 `(3,3)`, `(5,5)`, `(7,7)` → kernel sizes.
- 💾 `os.path.getsize()` → file size in bytes.
- ➗ `/1024` → convert bytes to KB.
- 🧠 If the question says **"Do not use library routine"**, manually implement that part.
- ✍️ Don't overcomplicate the code during the exam.
- 🔍 Read the exact question before deciding whether to use `cv2.imshow()` or `plt.subplot()`.

---

# 🎯 FINAL 5-MINUTE REVISION

```text
IMAGE READ
    ↓
imread()

GRAYSCALE
    ↓
cvtColor()

BINARY
    ↓
threshold()

NEGATIVE
    ↓
255 - pixel

LOG
    ↓
np.log()

GAMMA
    ↓
np.power()

HISTOGRAM
    ↓
equalizeHist()

COMPRESSION
    ↓
JPEG / PNG / WEBP

SMOOTHING
    ↓
filter2D()
GaussianBlur()
medianBlur()

EDGE DETECTION
    ↓
Laplacian()
```

---

# 💪 YOU ARE READY!

> 🧠 Understand the concept  
> ✍️ Remember the main function  
> 👀 Check the image path  
> ▶️ Run the code  
> 🔍 Check the output  
> 📝 Be ready to explain the important lines  
>
> **19 EXPERIMENTS → DONE ✅🔥**
>
> **Keep it simple. Stay calm. Read the question carefully.**
