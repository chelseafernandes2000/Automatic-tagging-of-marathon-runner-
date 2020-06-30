# Automatic-tagging-of-marathon-runner-
Image Text Detection, a complete project done using deep learning models like EAST Text detector and Tesseract OCR's OpenCV

<b>Abstract:</b>

In this project, by using deep learning and image processing we aim to create a system which will identify and recognize Bib number in  natural image collections associated with each marathon runner. A bib is typically a piece of durable paper or cardboard bearing a number as well as the event/sponsor logo that is attached to the body of a marathon runner. The bib has a unique bib number, printed in big fonts, along with the runner’s name and some other text. Our system will help the marathon managing staff to upload any natural image of the marathon and get the bib no. associated with that image. Along with pictures, a video of marathon can also be uploaded to get frames of images containing bib no.s as recognised by our system and all the information about the occurrence of a bib in the video will be stored in a downloadable file using OpenCV EAST text detector and tesseract-OCR and thus this can be used for real-time processing and can be tested through webcam feature.
We aim to also display relevant information which we can gather through images and videos of the marathon in the future. 


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


