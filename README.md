# 📦 Recommendation System - Phase 3

## Optimization, Scaling, and Final Evaluation

This repository contains two implementations of a Python-based Recommendation System developed for **Phase 3** of the project **Developing and Optimizing Data Structures for Real-World Applications**.
Both implementations demonstrate scalable user--user similarity recommendations with performance optimization.

------------------------------------------------------------------------

## 🔥 Project Overview

In this phase, major enhancements were made to improve efficiency and scalability, including:

-   **Optimized cosine similarity**
-   **Memory‑efficient dataset generation**
-   **Performance testing**
-   **Stress testing for scalability**
-   **Interactive and non‑interactive execution modes**

Both scripts fulfill all **Phase 3 requirements**:
**Optimization, scaling, and final evaluation.**

------------------------------------------------------------------------

## 📁 Files in this Repository

### **1. `optimized_rec_sys.py`**

A non‑interactive version used as the **baseline optimized engine**.\
It auto‑generates a dataset, computes recommendations, and runs a stress
test.

**Features:** - Fast execution\
- Clean output\
- Strong similarity results (dense dataset)\
- Useful for baseline comparison

------------------------------------------------------------------------

### **2. `ex1.py` (Interactive Version)**

A fully interactive recommendation system where the user chooses:

-   Number of users\
-   Number of items\
-   Sparsity level\
-   User ID\
-   Top‑K recommendations\
-   Whether to run a stress test\
-   Stress test dataset size

**Features:** - Dynamic dataset generation\
- Demonstrates scalability with large inputs\
- Real‑time performance measurement\
- Best suited for Phase 3 demonstration

------------------------------------------------------------------------

## 🚀 How to Run

### **Interactive Mode**

``` bash
python ex1.py
```

You will be prompted to enter dataset size, sparsity, Top‑K, and stress
test options.

### **Non‑Interactive Mode**

``` bash
python optimized_rec_sys.py
```

This runs automatically and prints: - Top recommendations\
- Stress test execution time

------------------------------------------------------------------------

## 📊 Example Outputs

### **Interactive Example**

    ===== INTERACTIVE RECOMMENDATION SYSTEM =====
    Enter number of users: 10500
    Enter number of items: 558
    Enter sparsity: 0.7

    Top-8 recommendations for user 345:
    User 5596 → 0.3559
    User 28 → 0.3487
    ...

    Stress test completed in 0.0064 seconds.

### **Non‑Interactive Example**

    Top recommendations for user 5:
    [(145, 0.9172), (10, 0.8887), (72, 0.8829), ...]

    Running stress test...
    Time: 0.1198

------------------------------------------------------------------------

## 🔧 Technologies Used

-   Python 3.x\
-   NumPy\
-   Time complexity analysis\
-   Memory‑optimized dataset generation\
-   Randomized user‑item matrix simulation

------------------------------------------------------------------------

## 🧪 Key Phase 3 Features Demonstrated

### ✔ **Optimization Techniques**

-   Faster cosine similarity\
-   Reduced memory footprint\
-   Removal of redundant loops

### ✔ **Scaling Strategy**

-   Dynamic dataset sizes\
-   High‑sparsity handling\
-   Stress testing for larger loads

### ✔ **Validation & Testing**

-   Edge‑case handling\
-   Stress test timing\
-   Progressive dataset scaling

### ✔ **Final Evaluation**

Both scripts demonstrate: - A baseline optimized model\
- A fully interactive, scalable model

Together, they fulfill all Phase 3 requirements.

------------------------------------------------------------------------

## 📘 Next Steps (Phase 4)

You are now ready to begin:

### **📄 Final Report**

A complete documentation of all phases.

### **📊 Presentation**

A polished summary of the system, optimization techniques, and results.

------------------------------------------------------------------------
