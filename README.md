# NTUEE_CommLab
NTUEE Communication Lab Final Project, Group 7, Quantum Image Processing - Quantum Edge Detection

* Intall requirement library
  
```shell script=
cd NTUEE_CommLab
pip install -r requirements.txt
```

* Run QHED algorithm on quantum simulator
```shell script=
python3 QHED_Simulator.py --name elephant --size 64 --noise --std 0.01
```

* You can use `examples/evaluate.ipynb` to evaluate the results. Three metrics are provided: MSE, PSNR, SSIM.

* Run QHED algorithm on IBM Quantum devices: Checkout `QHED_device.ipynb`. The code is able to perform image cropping, quantum circuit generation and transpilation as well as the aggregation of result from the quantum devices.