# MUFRocks
---
<p> This project focuses on providing automatic classification of satellite images of felsic, mafic, and ultramafic rocks from La Palmira and La Victoria, Colombia. This project uses two types of satellite images hand picked from the Earth Observation System (EOS). Namely, natural color (bands B04, B03, B02) and the infrared color vegetation (B08, B04, B03). The <b><i>Machine Learning algorithms</i></b> that were used in this study where the following: <b><i>Random Forest, K-Nearest Neighbors, Support Vector Machines, Logistic Regression, and Multilayer Perceptron</i></b>. Our results are as follows, the model generated with <b><i>K-Nearest Neighbors</i></b> performed best for <b><i>classifying natural color images</i></b>, with an <b><i>accuracy of 91%</i></b>, a <b><i>precision of 87%</i></b>, and a <b><i>recall of 88%</i></b>. <b><i>Random Forest</i></b> was the best model for <b><i>classifying infrared images</i></b> with an overall <b><i>accuracy of 83%</i></b>, a <b><i>precision of 31%</i></b>, and a <b><i>recall of 31%</i></b>. </p>
<hr>
<p> Files found within the OSGDAL folder belong to the <b><i>Natural Color</i></b> satellite bands, and are similar to the files found within the LLU_Colombia folder. The only exception being that the latter folder contains the clipped <b><i>Infrared Color Vegetation</i></b> satellite bands and a python file called Rock_data.py. This is the same file as the one found in the OSGDAL folder except with a name change, truth_data.py. </p>
<hr>
<p> The folder called Colombia_Geo contains the files that where provided to us by a Colombian university. These are files that can be accessed and viewed with OGR on any computer. These files contain a map of the area in La Victoria and Palmira with a set amount of rock specimiens already classified. </p>
<hr>
<p> As far as the ImagesUsed and InitialImages folders are concerned, InitialImages contains the initial images that were extracted from the EOS servers in both bands. While ImagesUsed contains the clipped images that were used in the training and loading aspect of this study. The main reason as to why the images where clipped is to lighten the load of the computational power and reduce the amount of time needed to complete the classification of the desired geological area. </p>
<hr>
<p> Finally, the neighbortest folder contains all the relevant Machine Learning algorithms used in this study identified by their initials. As well as the train and test csv files that were generated from the extracted and provided data. </p>
<hr>
<p> TLDR: 
* Learn/understand different technologies in order to analyze satellite images and their data
* Create a Python code to analyze satellite images and use Machine Learning to extract, train, and load all relevant information </p>
<hr>
