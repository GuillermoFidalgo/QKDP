# Quantum Key Distribution Protocol BB84
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
The program should:
1. use a random number generator at every point in the protocol where random numbers are needed,
2. be interactive allowing the user to run the program as many times as he/she likes and only stopping when the user wants to quit,
3. the user chooses if he wants Eve to be there or not,
4. the user chooses the length of the initial bit-string and how many bits will be used to verify the security of the key,
5. the program must be fully commented,
6. along with the key produced, the program should calculate all relevant statistics such as probability that the key is compromised, percentage of discarded bits and percentage of bits that Eve gets. These are just some examples, I'm sure you can think of more.


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


# Disclaimer
Depending on the parameters used for the simulation of BB84 this may take a long time. Please be patient in order to see the results.

If you use Google Colab, then the section that uses a CSV file from a more time consuming simulation **will not work as is**.
Please change the argument in

```
df=pd.read_csv('Distribution-Data-for-BB84.csv')
```
to
```
df=pd.read_csv('https://raw.githubusercontent.com/GuillermoFidalgo/QKDP/master/Distribution-Data-for-BB84.csv')
```

All images produced in this script are downloaded to memory in a folder called `plots_BB84`


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://guillermofidalgo.github.io"><img src="https://avatars.githubusercontent.com/u/17858942?v=4?s=100" width="100px;" alt="Guillermo A. Fidalgo-Rodríguez"/><br /><sub><b>Guillermo A. Fidalgo-Rodríguez</b></sub></a><br /><a href="#content-GuillermoFidalgo" title="Content">🖋</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!


<!-- Adding dummy commit -->