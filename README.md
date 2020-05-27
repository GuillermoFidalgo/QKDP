# Quantum Key Distribution Protocol BB84
The program should:  
1. use a random number generator at every point in the protocol where random numbers are needed, 
2. be interactive allowing the user to run the program as many times as he/she likes and only stopping when the user wants to quit, 
3. the user chooses if he wants Eve to be there or not, 
4. the user chooses the length of the initial bit-string and how many bits will be used to verify the security of the key, 
5. the program must be fully commented, 
6. along with the key produced, the program should calculate all relevant statistics such as probability that the key is compromised, percentage of discarded bits and percentage of bits that Eve gets. These are just some examples, I'm sure you can think of more.

**Disclaimer**
Depending on the parameters used for the simulation of BB84 this may take a long time. Please be patient in order to see the results.

## Running the notebook using Google Colab (online)

To run the code in Google Colab you just go the [notebook](https://github.com/GuillermoFidalgo/QKDP/blob/master/BB84.ipynb) and click on the <img src="https://github.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/blob/master/colab-button.png" alt="Open In Colab" width="120"/> found on top of the file and you will instantly open the document online. 


## Running the code locally 
You can either download or fork this repository

In case you fork you can do 

```shell
cd ~/YOUR/OWN/DIRECTORY  # Choosing a destination for code
git clone https://github.com/[YOUR_USERNAME]/QKDP.git  # download from your fork
cd QKDP  
```


In order to use the code with Juyper Notebooks or Jupyter Lab do one of the following

Open the terminal of your choice (if you have Anaconda open Anaconda prompt and go to the destination folder in the step before)
- Jupyter Notebook

```shell
jupyter notebook --no-browser
```

- Jupyter Lab

```shell
jupyter lab --no-browser
```


In both cases wait for output, copy and paste this into your browser of choice
 should be something like 
```shell
# Or copy and paste one of these URLs:   
# http://localhost:8888/?token=13214e004a5f1b60fc2971244baa8292cfafa5d65973c6fe    
# or http://127.0.0.1:8888/?token=13214e004a5f1b60fc2971244baa8292cfafa5d65973c6fe
```


## Dependencies
1. Must have Python and Jupyter Installed (Anaconda Distribution is easiest to get)
2. numpy
3. matplotlib
4. pandas


