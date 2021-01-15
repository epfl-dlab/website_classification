# Semester projet: language-independent websites embeddings

This repository contains the code supporting the semester project. The project report can be found [here](websites_embedding.pdf). The code is separated in folders, depending on its scope. 

The most important files are the following, listed in the order to use them for computing websites embeddings:

- textual/**collect_html.py** to retrieve the HTML responses from a list of urls and store them in a json file
- graphical/**collect_screenshots.py** to take screenshots from a list of urls
- graphical/**resnet_train.py** to train a computer vision model (CVM) from screenshots
- graphical/**compute_visual_embeddings.ipynb** to get the visual embeddings using the trained CVM
- final/**get_textual_embeddings.py** to get the textual embedding from the HTML responses
- final/**final_model.ipynb** for joining the textual and visual embeddings and train classifiers 

The required libraries and their version are:
- beautifulsoup 4.9.3
- sentence-transformers 0.3.9
- pytorch 1.5.0
- torchvison 0.6.0
- selenium 3.141.0
