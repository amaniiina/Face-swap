# Face-swap

This program takes two pictures and swaps parts of them depending on the given mask image.
it is implemented using an image pyramid, wihtout using OpenCV functions and instead by creating the Laplacein pyramid using Transform-Fourier.

Input: pic1.png, pic2.png, mask.png (The mask is created manually by Paint or any other software and inputted to the program)\
Output: the two pictures with swapped parts
