# Automatic-tagging-of-marathon-runner-
Image Text Detection, a complete project done using deep learning models like EAST Text detector and Tesseract OCR's OpenCV

<b> 1. Main.py: </b><br>
This is the main file that consists of main code of functionality. 

<b> 2. templates: </b><br>
This is folder contains all html pages used for taking inputs and rendering outputs with the help of flask.

<b> 3. static: </b><br>
This folder is used to store all the images and video files upload by user.

<b> 4. 4digit: </b><br>
This folder consists some 4-digit bib numbered images for testing.

Documentation.docx is the official documentation of the project and Automatic tagging of Marathon Runners.pptx is the presentation to understand the concepts more deeply.

Instructions after you've cloned the repo on your computer:

1. Pip Install all that is imported

2. Note that in filtertext() the regex is currently only taking images with H,F,M and thus will not read any other format of numbers and so you can change the regex as per your convinience 

3. Make sure to put all paths in the program are as per where the files are in your pc

The project uses the front end framework Flask so, To run flask:

1.Go to CMD, go to the folder of your python code
2. Run the python file through cmd
3. Use the URL as given to open it in any browser
4. Do not close CMD if you want the project to be running

For more details and to see screenshots of working model, check Documentation.docx


