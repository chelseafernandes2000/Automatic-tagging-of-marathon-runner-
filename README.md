# Automatic-tagging-of-marathon-runner-
Image Text Detection

Using deep learning models like EAST text detector and Tesseract OCR we implemented an image text detector for detecting bib no.s associated with pictures and videos of marathon runners

Check Project branch for all the files

Instructions after you've downloaded the file on your computer:

1. Pip Install all that is imported

2. Download frozen_east_text-detector.pb from https://www.kaggle.com/yelmurat/frozen-east-text-detection#frozen_east_text_detection.pb and put it in the same folder as that of the  program

3. Note that in filtertext() the regex is currently only taking images with H,F,M and thus will not read any other format of numbers and so you can change the regex as per your convinience 

4. Make sure to put all paths in the program are as per where the files are in your pc

The project uses the front end framework Flask so, To run flask:

1.Go to CMD, go to the folder of your python code
2. Run the python file through cmd
3. Use the URL as given to open it in any browser
4. Do not close CMD if you want the project to be running

For more details and to see screenshots of working model, check Ducumentation
