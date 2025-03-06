### Guideline 
- every **py script/file** should be run from the **home directory of the project**. Every path defined should start from this path. Use `os.getcwd()` to and then add any path to it. For example, to get to the path `src/biane.pdf` in project directory `gene-disease-graph`, make sure to run script from project directory and use `os.path.join(os.getcwd(), "src", "biane.pdf")`.
- import should be managed taking in account the above assumption.

### Task *(deadline - 27th Feb)*
- make new directory `utils` to inside that make new file `graph_loader.py`. It should ve a class called `GraphLoader` which takes in **input path of the file** and process it and convert it in a `DGL graph` and **save it to output path**.
- `graph_laoder.py` should also have a `NameEncoder` class which will convert codename of disease and gene to `int` basically mapping of mame to integer. And also do the reverse.
- while reading the data from file, we can get entire list of **different num of diseases and genes**. For one-hot-encoding it is important to assign every node (disease and gene) a **unique integer**.

### EDA 
#### from dgl object to gml file (python script)
1. EDA -> grephi -> 
2. Dgl / networkx -> jupyter 
screenshots -> src/eda/[`.ipynb`, images, video, animation]