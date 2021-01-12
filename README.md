# Language-independent websites embeddings

This repository contains the code supporting the semester project semester. The project report can be found under ??. The code is separated in folders, depending on its scope. 

The most important files are the following:

- textual/**collect_html.py** to retrieve the HTML responses from a list of urls and store them in a json file
- graphical/**collect_screenshots.py** to take screenshots of websites from a list of urls
- graphical/**resnet_train.py** to train a computer vision model (CVM) from screenshots of websites
- graphical/**compute_visual_embeddings.ipynb** to embed the screenshots of websites using the trained CVM
- final/**get_textual_embeddings.py** to get the textual embedding from the HTML responses
- final/**final_model.ipynb** for joining the textual and visual embeddings of websites and train classifiers 

The required libraries and their version are:
- beautifulsoup 4.9.3
- sentence-transformers 0.3.9
- pytorch 1.5.0
- torchvison 0.6.0
- selenium 3.141.0
