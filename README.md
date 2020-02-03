# Molecular_ML
Machine learning for properties of chemical compounds.  

Valid model:  
Gaussian Process Regression using graph kernel


Edit config.py for different purpose  

Prepare 
```
git clone https://github.com/XiangyanSJTU/Molecular_ML.git
git clone https://github.com/sungroup-sjtu/AIMS_Tools.git # change to dev-xy branch
git clone https://github.com/yhtang/GraphDot.git
```
GPR
```
cd Molecular_ML/run
python3 GPR.py -i result-ML-density.txt -p density
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

