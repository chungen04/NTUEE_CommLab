# Importing standard Qiskit libraries and configuring account
from qiskit import *
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')

from PIL import Image
import math
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="QHED simulator")
    parser.add_argument('--name', type=str, default='elephant', choices=['elephant', 'cameraman'])
    parser.add_argument('--size', type=int, default=64, help="crop size")
    parser.add_argument('--noise', action='store_true', help="With Noise or not")
    parser.add_argument('--std', default = 0.01, type=float)
    
    return parser.parse_args()
# python3 QHED_Simulator.py --name elephant --size 64 --noise --std 0.01

args = get_args()

crop_size = args.size
name = args.name # [elephant, cameraman]

### Noise
Noise = args.noise
mean = 0
std = args.std



def plot_image(img, i, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
    
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='binary')
    plt.savefig(f"{save_dir}/{title}_{i}.jpg")
    plt.close()
    # plt.show()
    
def addNoise(img, mean, std, Noise = False):
    if Noise == False:
        return img
    else:
        img = np.array(img/255.0, dtype=float)
        noise = np.random.normal(mean, std, img.shape)
        out = img + noise
        
        out = np.clip(out, 0.0, 1.0)
        out = np.uint8(out*255)
        
        return out
    
def amplitude_encode(img_data):
    
    # Calculate the RMS value
    rms = np.sqrt((np.sum(np.square(img_data, dtype='float64'), axis=None)))
    
    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)
    # Return the normalized image as a numpy array
    return np.array(image_norm)

def generateCircuit(image_norm_h, image_norm_v):
    
    # Create the circuit for horizontal scan
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(image_norm_h, range(1, total_qb))
    qc_h.h(0)
    qc_h.unitary(D2n_1, range(total_qb))
    qc_h.h(0)
    # display(qc_h.draw('mpl', fold=-1))

    # Create the circuit for vertical scan
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(image_norm_v, range(1, total_qb))
    qc_v.h(0)
    qc_v.unitary(D2n_1, range(total_qb))
    qc_v.h(0)
    # display(qc_v.draw('mpl', fold=-1))

    # Combine both circuits into a single list
    circ_list = [qc_h, qc_v]
    
    return qc_h, qc_v, circ_list

def QuantumSimulate(qc_h, qc_v, circ_list):
    # Simulating the cirucits
    back = Aer.get_backend('statevector_simulator')
    results = execute(circ_list, backend=back).result()
    sv_h = results.get_statevector(qc_h)
    sv_v = results.get_statevector(qc_v)

    from qiskit.visualization import array_to_latex
    print('Horizontal scan statevector:')
    # display(array_to_latex(sv_h[:30], max_size=30))
    print()
    print('Vertical scan statevector:')
    # display(array_to_latex(sv_v[:30], max_size=30))
    
    return sv_h, sv_v

def scanEdge(sv_h, sv_v):
    edge_scan_h = np.array([sv_h[f'{2*i+1:0{total_qb}b}'].real for i in range(2**data_qb)]).reshape(image.shape[1], image.shape[2])
    edge_scan_v = np.array([sv_v[f'{2*i+1:0{total_qb}b}'].real for i in range(2**data_qb)]).reshape(image.shape[1], image.shape[2]).T

    return edge_scan_h, edge_scan_v



img_dir = f"imgs/{name}.png"

data_qb = int(math.log2(crop_size**2))
anc_qb = 1
total_qb = data_qb + anc_qb
print(f"Total Qubits: {total_qb}")

noise = f"Noise_std{std}" if Noise else ""
save_dir = f"result_{name}/result_image_crop{crop_size}{noise}"
os.makedirs(save_dir, exist_ok=True)

image = np.array(Image.open(img_dir).convert('L'))
print(f"Original Image Shape: {image.shape}")

image = addNoise(image, mean, std, Noise = Noise)

original_img = [image.shape[0], image.shape[1]]
times = int(image.shape[0]/crop_size)
crop_image = np.zeros((times**2, crop_size, crop_size))
for i in range(times):
    for j in range(times):
        crop_image[i*times+j] = image[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
image = crop_image

print(f"Cropped Image Shape: {image.shape}")


    # Horizontal: Original image
image_norm_h = np.asarray([amplitude_encode(image[i]) for i in range(image.shape[0])])

# Vertical: Transpose of Original image
image_norm_v = np.asarray([amplitude_encode(image[i].T) for i in range(image.shape[0])])
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)


Total_scan_h = []
Total_scan_v = []
for i in range(image_norm_h.shape[0]):
    print(f"Idx: {i}")
    qc_h, qc_v, circ_list = generateCircuit(image_norm_h[i], image_norm_v[i])
    sv_h, sv_v = QuantumSimulate(qc_h, qc_v, circ_list)
    edge_scan_h, edge_scan_v = scanEdge(sv_h, sv_v)
    Total_scan_h.append(edge_scan_h) #[16, 128, 128]
    Total_scan_v.append(edge_scan_v)
    plot_image(edge_scan_h, i=i, title='Horizontal')
    plot_image(edge_scan_v, i=i, title='Vertical')
    

Total_scan_h = np.asarray(Total_scan_h)   
Total_scan_v = np.asarray(Total_scan_v)
Result_h = np.zeros((original_img[0], original_img[1]))
Result_v = np.zeros((original_img[0], original_img[1]))

for i in range(times):
    for j in range(times):
        Result_h[crop_size*i:crop_size*(i+1), crop_size*j:crop_size*(j+1)] = Total_scan_h[i*(times)+j,:,:] 
        Result_v[crop_size*i:crop_size*(i+1), crop_size*j:crop_size*(j+1)] = Total_scan_v[i*(times)+j,:,:]

plot_image(Result_h, i=i, title='Result_Horizontal')
plot_image(Result_v, i=i, title='Result_Vertical')

npy_savedir = f"result_{name}/result_npy"
os.makedirs(npy_savedir, exist_ok=True)
np.save(os.path.join(npy_savedir, f"Result_h_{crop_size}{noise}.npy"), Result_h)
np.save(os.path.join(npy_savedir, f"Result_v_{crop_size}{noise}.npy"), Result_v)
