# Molecular_ML
Machine learning for properties of chemical compounds.  

Valid model:  
Gaussian Process Regression using graph kernel


Edit config.py for different purpose  

Prepare 
```
git clone https://github.com/XiangyanSJTU/Molecular_ML.git
```

to be finished.

GPR
```
cd Molecular_ML/run
python3 GPR.py 
```
tSNE analysis
```
cd scripts
python3 tSNE.py -i cp.txt -p cp -t graph_kernel
```
property-heavy_atoms relationship
```
cd scripts
python3 NheavyPropertyAnalysis.py -i cp.txt -p cp
```

