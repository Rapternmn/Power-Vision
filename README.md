# Power-Vision

Primary focus was to detect a page, apply four-point transform method to get an entire page and remove the background, identify multiple columns in the page by applying morphological transforms(Erosion + Dilation) , crop the images and pass them sequentially into pytesseract ocr to get an appropriate text output. Finally converting text to speech.

Example :
-Find a test image : half.jpg
-Find a sequence of cropped-output images in the folder "Crop Outputs"
It shows the accuracy of boundary detection and cropping accuracy+sequencing of the images

Demo video : https://www.youtube.com/watch?v=CcR5tph-pm4