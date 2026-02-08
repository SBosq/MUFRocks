# MUFRocks — Automated Geological Classification Using Satellite Imagery

MUFRocks is a Python-based project that combines **GIS, satellite imagery, and machine learning** to classify felsic, mafic, and ultramafic rock formations in Colombia. The project features **reproducible data pipelines**, efficient preprocessing of natural color and infrared imagery, integration of geospatial reference data, and ML models achieving up to **91% classification accuracy**. It demonstrates practical experience in **Python automation, geospatial workflows, and reproducible analytical pipelines**, aligning closely with modern CI/CD and ArcGIS API practices.

---

## Overview
MUFRocks is focused on the automated classification of rock formations using satellite imagery from La Palmira and La Victoria, Colombia. The project integrates geospatial reference data and Earth Observation System (EOS) imagery to build reproducible pipelines for **data preparation, model training, and evaluation**.

---

## Data Sources
- **Satellite Imagery:**  
  - Natural color bands (B04, B03, B02)  
  - Infrared vegetation bands (B08, B04, B03)  
  - Retrieved from the Earth Observation System (EOS)
- **Geospatial Reference Data:**  
  - Ground-truth rock classification data provided by a Colombian university  
  - Stored in OGR-compatible formats for portability and spatial accuracy

---

## Methodology
1. **Data Preparation**
   - Extracted and clipped satellite imagery to reduce computational load and improve processing efficiency.
   - Aligned raster data with vector-based geospatial reference files.
   - Standardized datasets for consistent training and evaluation.

2. **Machine Learning Models**
   - Random Forest  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machines (SVM)  
   - Logistic Regression  
   - Multilayer Perceptron (MLP)

3. **Training & Evaluation**
   - Generated structured training and testing datasets.
   - Evaluated models using **accuracy, precision, and recall**.
   - Maintained clear separation of data preparation, training, and evaluation steps to support reproducibility.

---

## Results
- **Natural Color Imagery**
  - Best model: **K-Nearest Neighbors**
  - Accuracy: **91%**
  - Precision: **87%**
  - Recall: **88%**

- **Infrared Imagery**
  - Best model: **Random Forest**
  - Accuracy: **83%**
  - Precision: **31%**
  - Recall: **31%**

---

## Project Structure (High-level)
- `OSGDAL/` – Processing pipelines for natural color imagery  
- `LLU_Colombia/` – Infrared imagery workflows and Python processing scripts  
- `Colombia_Geo/` – OGR-compatible geospatial reference data  
- `ImagesUsed/` – Clipped imagery used for model training and evaluation  
- `InitialImages/` – Raw satellite imagery retrieved from EOS  
- `neighbortest/` – Machine learning models and generated train/test datasets  


---

## Technologies Used
- **Programming:** Python  
- **GIS & Geospatial:** GDAL, OGR, QGIS, EOS imagery  
- **Machine Learning:** scikit-learn  
- **Data Formats:** Raster imagery, CSV, OGR-compatible vector data  

---

## Key Takeaways
- Built **reproducible Python pipelines** for geospatial data processing.  
- Applied machine learning techniques to real-world satellite imagery.  
- Optimized workflows for efficiency and maintainability.  
- Strengthened understanding of GIS tooling, data integrity, and documentation practices.  

---

## Future Improvements
- Incorporate additional satellite imagery and geospatial data (shapefiles) from future field visits.  
- Validate classifiers with field data to ensure real-world accuracy.  
- Combine macro-level satellite analysis with micro-level field sample analysis for more robust geological classification.  
- Explore hyperspectral drone imagery to increase data resolution and enhance pipeline insights.  
- Continue improving **reproducible Python workflows** for data processing, model training, and evaluation.  
