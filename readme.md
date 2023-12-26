### Team members:

## Joël
## Hajar Lachheb
## Saurabh Nagarkar
## Miquel Sugrañes Pàmies

### Running script for the first time

# Open project’s root folder in terminal:
 ```bash
cd <root_folder_of_project>/ 
```

# Create virtual environment: 
 ```bash
python3 -m venv venv/ 
```

# Open virtual environment:
 ```bash
source venv/bin/activate 
```

# Install required dependencies: 
```bash
pip install -r requirements.txt 
```

# Close virtual environment: 
```bash
deactivate
```

## Execute scripts
# Open folder in terminal: 
```bash
cd <root_folder_of_project>/
 ```

# Open the virtual environment:
```bash
source venv/bin/activate
```

# Fuzzy C Means algorithm for all datasets:
```bash
python3 main_fuzzycmeans.py
 ```

# Agglomerative clustering algorithm for vowel and pen-based datasets:
```bash
python3 main_agglomerativeclustering.py
 ```

# Agglomerative algorithm for adult dataset (!! It is not recommended due to comp. cost):
```bash
python3 main_agglomerativeclustering_adult.py
 ```

# Mean shift algorithm for all datasets:
```bash
python3 main_meanshift.py
 ```

# Bisecting K Means for all datasets: 
```bash
python3 main_Bisecting Algorithm.py
 ```

# Close virtual environment:
 ```bash
deactivate
```
