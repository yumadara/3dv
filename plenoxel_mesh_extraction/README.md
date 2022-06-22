This script should be used with the same conda environment used for plenoxels.

- Additionaly, one needs to install pyvista: 

conda install -c conda-forge pyvista
- Put mesh_exporter.py inside the opt directory (svox2/opt)
- change line 723ff of svox2/svox2/svox2.py

From:
```
l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
```
To:
```
l[:, 0] = torch.clamp_max(l[:, 0].float(), gsz[0] - 2)
l[:, 1] = torch.clamp_max(l[:, 1].float(), gsz[1] - 2)
l[:, 2] = torch.clamp_max(l[:, 2].float(), gsz[2] - 2)
```
- run script

How to run the script:

```
python mesh_exporter.py <path to ckpt.npz file> <resoliton of the mesh>
```
E.g.:
```
python mesh_exporter.py ../checkpoints/ckpt_syn/256_to_512_fasttv/lego/ckpt.npz 0.2
```

The checkpoint for the synthetic lego can be found [here](https://drive.google.com/drive/folders/1rVqMxlQrWXklOAlQ--NdpiXrNcAwNORP?usp=sharing).
I placed for e.g. in /svox2/checkpoints

Currently I am running wit resolution 0.2, which is very low. It would be good to try with 0.5 and even 1 is possible

-Output:

The script creates a ckpt.obj and ckpt_colored.obj file in the same folder og the ckpt.npz file.