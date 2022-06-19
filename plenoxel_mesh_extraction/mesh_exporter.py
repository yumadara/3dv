import numpy as np
import pyvista as pv
import svox2, mcubes, torch, numpy, argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    args = parser.parse_args()
    targetpath = args.ckpt[:-3] + "obj"
    print("result will be saved to:", targetpath)
    print("loading sparse grid")
    grid = svox2.SparseGrid.load(args.ckpt)

    resfactor = 0.1#1  # increase/decrease to get higher or lower resolution meshes. Be very careful when increasing.
    # 0.3 is nice for previewing the rough stuff

    resx = int(grid.shape[0] * resfactor)
    resy = int(grid.shape[1] * resfactor)
    resz = int(grid.shape[2] * resfactor)

    densitygrid = numpy.zeros((resx, resy, resz))
    print("converting densities to non-sparse numpy array")
    #not exactly fast, but not excrutiating slow either.
    for x in range(resx):
        print("Progress:%i %%" % (100 * x / resx))
        xvals = [
                    -0.5 + x / resx] * resx  # an array with x-size size and filled with current x value, by default ranges -.5 to .49 something
        xvals = numpy.array(xvals)
        for y in range(resy):
            yvals = [-0.5 + y / resy] * resy  # an array with y-size size and filled with current y value
            yvals = numpy.array(yvals)
            zvals = numpy.array(range(0, resz)) / resz
            zvals = zvals - 0.5
            samplepos = numpy.dstack([xvals, yvals, zvals])[0]  # get a full z-line at x,y coords worth of 3d coordinates.
            #u = grid.sample(torch.Tensor(samplepos), want_colors=False)[0]  # only get densities, no colors
            d_and_c = grid.sample(torch.Tensor(samplepos), want_colors=False)  # only get densities, no colors
            u = d_and_c[0]
            v = u.detach().numpy()  # turn tensor into numpy array.
            vt = v.T[0]  # turn array from shape (128,1) to (1,128)
            densitygrid[x, y] = vt  # feed the extracted line into our density array

    densitygrid[densitygrid<0.05]=0
    v, t = mcubes.marching_cubes(densitygrid, 20)  # adjust value to your scene. start with 0.
    mcubes.export_obj(v, t, targetpath)
    normals = estimate_normals(targetpath)
    add_color(targetpath, v, normals, grid)


def estimate_normals(targetpath):

    mesh = pv.read(targetpath)
    mesh_w_no = mesh.compute_normals(cell_normals=False)
    normals = mesh_w_no['Normals']
    #print(normals.shape)
    return normals


def add_color(targetpath, v, normals, grid):
    c_fn = targetpath[:-4]+"_colored.obj"
    # points might be in wrong location
    vw = grid.grid2world(torch.from_numpy(v).to(torch.float32))#TODO do this tranformation to wolrd coordinates?

    origins = vw + torch.from_numpy(10*normals).to(torch.float32)# a bit above
    # normal might have to point in oder direction
    dirs = torch.from_numpy(-normals).to(torch.float32)
    rays = svox2.Rays(origins, dirs)

    out_rgb1 = grid.volume_render(rays)

    N_v = len(v)
    with open(c_fn, 'w') as outfile, open(targetpath, 'r', encoding='utf-8') as infile:
        counter = 0
        for line in infile:
            if counter < N_v:
                col = out_rgb1[counter].cpu().detach().numpy()
                new_line = line[:-1] + f" {col[0]} {col[1]} {col[2]}\n"
                outfile.write(new_line)
            else:
                outfile.write(line)
            counter += 1


if __name__ == "__main__":
    main()