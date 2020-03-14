# Image Classification for a City Dog Show
In this project you will use a created image classifier to identify dog breeds. 
## Description

Your city is hosting a citywide dog show and you have volunteered to help the organizing committee with contestant registration. Every participant that registers must submit an image of their dog along with biographical information about their dog. The registration system tags the images based upon the biographical information.

Some people are planning on registering pets that aren’t actual dogs.

You need to use an already developed Python classifier to make sure the participants are dogs.

## Tasks
- Using `Python` skills, determine which image classification algorithm works the **"best"** on classifying images as "dogs" or "**not** dogs".
  
- Determine how well the "best" classification algorithm works on correctly identifying a dog's breed. You can see the image classifier as a tool that has an input and an output. The Input is an image. The output determines what the image depicts. (for example: a dog). Be mindful of the fact that image classifiers do not always categorize the images correctly.
  
- Time how long each algorithm takes to solve the classification problem. With computational tasks, there is often a trade-off between accuracy and runtime. The more accurate an algorithm, the higher the likelihood that it will take more time to run and use more computational resources to run.

## Program Outline

- Time your program
    - Use Time Module to compute program runtime
- Get program Inputs from the user
    - Use command line arguments to get user inputs
- Create Pet Images Labels
    - Use the pet images filenames to create labels
    - Store the pet image labels in a data structure (e.g. dictionary)
- Create Classifier Labels and Compare Labels
    - Use the Classifier function to classify the images and create the classifier labels
    - Compare Classifier Labels to Pet Image Labels
    - Store Pet Labels, Classifier Labels, and their comparison in a complex data structure (e.g. dictionary of lists)
- Classifying Labels as "Dogs" or "Not Dogs"
    - Classify all Labels as "Dogs" or "Not Dogs" using dognames.txt file
    - Store new classifications in the complex data structure (e.g. dictionary of lists)
- Calculate the Results
    - Use Labels and their classifications to determine how well the algorithm worked on classifying images
- Print the Results

## Running a Program using command line arguments

- To run a program like `check_images.py` with defualt argument use:
    ``` 
    python check_images.py 
    ``` 
- To run a program like `check_images.py` using the command line argument `--dir`, first open a terminal window within the Project Workspace. Next type the following and hit enter to run the program (this example - `check_images.py`). Notice that all command line arguments are specified after the program name (this example - `check_images.py`) and they are indicated by the `--` that proceeds their variable name (this example : `dir`) with the value following the variable name (in this example the string : pet_images/).
    ``` 
    python check_images.py --dir pet_images/
    ``` 
- Another example of call
    ``` 
    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
    ``` 
## Testing the classifier function
Test the environment and the classifier function that we will be using to classify the pet images. This function is located in the classifier.py program. Test your environment by running the `test_classifier.py`: 
``` 
python test_classifier.py 
``` 
# Batch Processing

Now after completing coding `check_images.py`, we are ready to run it on all 3 models. One way to do this is to call the program from the terminal window for one of the models, wait until it completes running, record it's results, and then repeat for the other two models.

An easier way to handle this task is with **batch processing** using a **shell script**. This is provided in program run_models_batch.sh within the workspace. Here is the code below: 
``` 
#  Code from run_models_batch.sh 
python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt
     > resnet_pet-images.txt
python check_images.py --dir pet_images/ --arch alexnet  --dogfile dognames.txt  
     > alexnet_pet-images.txt
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt 
     > vgg_pet-images.txt
``` 

You will also notice that each file ends with `> filename.txt`. The `>` is a **pipe** and it pipes the output from the console into a file. The file contains the filename of the model being used. This way after each run, the results are automatically stored in your workspace.

To run file run_models_batch.sh in the workspace, open a terminal window (in Unix/Linux/OSX/Lab Workspace) and type the following:
``` 
sh run_models_batch.sh
``` 

If you want to batch process the program on a Windows computer you will need to follow the instructions found [here](https://github.com/udacity/AIPND/blob/master/notes/lab_intro-to-python-lab.md#running-batch-files-on-windows-os-locally).

The result file are used to compare between the three architect to find which onne is the best.


## Important Notes:

For this image classification task you will be using an image classification application using a deep learning model called a convolutional neural network (CNN). CNNs work particularly well for detecting features in images like colors, textures, and edges; then using these features to identify objects in the images. You'll use a CNN that has already learned the features from a giant dataset of 1.2 million images called [ImageNet]. There are different types of CNNs that have different structures (architectures) that work better or worse depending on your criteria. With this project you'll explore the three different architectures (**AlexNet**, **VGG**, and **ResNet**) and determine which is best for your application.

`classifier.py`contains *classifier* function that allow to use these CNNs to classify images. The `test_classifier.py` file contains an example program that demonstrates how to use the *classifier* function. This project, focus on using your Python skills to complete these tasks using the *classifier* function.

Remember that certain breeds of dog look very similar. The more images of two similar looking dog breeds that the algorithm has learned from, the more likely the algorithm will be able to distinguish between those two breeds. We have found the following breeds to look very similar: [Great Pyrenees] and [Kuvasz], [German Shepherd] and [Malinois], [Beagle] and [Walker Hound], amongst others.




[ImageNet]:<http://www.image-net.org/>
[Great Pyrenees]:<https://www.google.com/search?q=Great+Pyrenees&source=lnms&tbm=isch&sa=X&ved=0ahUKEwje252-kpfZAhVF3FMKHeXwB3IQ_AUICigB&biw=1112&bih=1069>

[Kubasz]:<https://www.google.com/search?tbm=isch&q=Kuvasz&spell=1&sa=X&ved=0ahUKEwi9_9fTkpfZAhWB7FMKHXlKDWoQBQg6KAA&biw=1112&bih=1069&dpr=1>

[German Shepherd]:<https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=d7F8WpaaMc_VzgLW8LvABw&q=German+Shepherd&oq=German+Shepherd&gs_l=psy-ab.3..0i67k1j0l2j0i67k1j0l6.31751.41069.0.41515.29.18.4.7.9.0.131.1164.14j2.17.0....0...1c.1.64.psy-ab..2.26.1140.0..0i10k1j0i13k1.112.xUB8_AoVF9w>

[Malinois]:<https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=orF8WtHWDcOdzwLnyLXgBw&q=Malinois&oq=Malinois&gs_l=psy-ab.3..0l3j0i67k1l3j0l2j0i67k1j0.31864.42125.0.42493.23.20.0.1.1.0.132.1460.14j4.19.0....0...1c.1.64.psy-ab..8.14.926.0...75.U5aOu6JZ9Vk>

[Beagle]:<https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=zbF8WqTiHZDxzgKlm5SYBw&q=Beagle&oq=Beagle&gs_l=psy-ab.3..0i67k1j0l2j0i67k1l2j0l5.29396.33482.0.34041.12.8.3.1.1.0.126.585.6j2.8.0....0...1c.1.64.psy-ab..0.12.609...0i10k1.0.Dr92CW2Kqqo>

[Walker Hound]:<https://www.google.com/search?biw=1112&bih=1069&tbm=isch&sa=1&ei=8LF8WteAGND0zgKvlL-IBw&q=Walker+hound&oq=Walker+hound&gs_l=psy-ab.3..0l10.20697.23454.0.23773.12.10.0.2.2.0.81.601.10.10.0....0...1c.1.64.psy-ab..0.12.610...0i67k1.0.GI0QxI1sadY>