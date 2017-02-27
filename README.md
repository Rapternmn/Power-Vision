# Power-Vision


Primary goal was to enable Blind people to analyze text content via audio outputs

We applied following methodology to achive this goal :

1) Detect and extract a page by applying four-point transform method 

2) Identify multiple columns in the page by applying morphological transforms(Erosion + Dilation)

3) Crop the images and pass them sequentially into pytesseract ocr to get an appropriate text output. 

4) Converting text to speech.

Example :

-Find a test image : half.jpg

-Find a sequence of cropped-output images in the folder "Crop Outputs"

It shows the accuracy of boundary detection and cropping accuracy+sequencing of the images

Demo video : https://www.youtube.com/watch?v=CcR5tph-pm4