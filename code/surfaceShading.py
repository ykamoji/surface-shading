import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import normalize
import os

from displaySurfaceNormals import displaySurfaceNormals

image_height = 120
image_width = 120

def read_images(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def getSurface(I, L):
    g = np.linalg.solve(L.T @ L, L.T @ I)
    g = normalize(g, axis=0)
    # g = np.linalg.norm(g, axis=1)
    return g.T

def prepare_data(filepath):
    ImArray = []
    lightDirs = []
    nImages = 0

    with open(filepath + '/LightSource.txt', 'r') as f:
        for line in f.readlines():
            nImages += 1
            line = line.strip()
            point = list(map(int, line[line.find('(') + 1: line.find(')')].split(',')))
            lightDirs.append(np.array(point).astype(np.float32))

    lightDirs = normalize(lightDirs, axis=1)
    # lightDirs = np.linalg.norm(lightDirs, axis=0)

    for i in range(1, nImages + 1):
        pic = read_images(filepath + '/pic' + str(i) + '.bmp')
        ImArray.append(pic.ravel())

    ImArray = np.asarray(ImArray)

    return ImArray, lightDirs

def plot_surface_normals(N):
    surfaceNormals = np.copy(np.reshape(N, (image_height, image_width, 3)))
    surfaceNormals = (surfaceNormals + 1.0) / 2.0
    displaySurfaceNormals(surfaceNormals, 'Extra')

def reflectance_map(heightMap, path):
    heightMap = np.copy(np.reshape(heightMap, (image_height, image_width)))
    plt.figure()
    plt.imshow(heightMap)
    plt.colorbar(label='Distance to Camera')
    plt.title('Height map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.savefig(path+'/reflectanceMap')


def save_point_cloud(Z, filepath):
    Z_map = np.reshape(Z, (image_height, image_width)).copy()
    data = np.zeros((image_height * image_width, 3), dtype=np.float32)
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_height):
        for j in range(image_width):
            idx = i * image_width + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_height - 1 - i][j]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)

def open_point_cloud(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

def sparse_multiplcation(sparse_matrix, dz, surfaceNormals, image):
    non_empty_pixels = np.argwhere(image != 0)
    meshgrid = np.zeros((image_height, image_width), dtype=np.int16)

    for i, (h, w) in enumerate(non_empty_pixels):
        meshgrid[h, w] = i

    for i, (h, w) in enumerate(non_empty_pixels):

        n_x, n_y, n_z = surfaceNormals[h, w]

        j = i * 2
        if image[h, w + 1]:
            dz[j] = -n_x / n_z
            k = meshgrid[h, w + 1]
            sparse_matrix[j, k] = 1
            sparse_matrix[j, i] = -1

        elif image[h, w - 1]:
            dz[j] = -n_x / n_z
            k = meshgrid[h, w - 1]
            sparse_matrix[j, k] = -1
            sparse_matrix[j, i] = 1

        j = i * 2 + 1
        if image[h + 1, w]:
            dz[j] = -n_y / n_z
            k = meshgrid[h + 1, w]
            sparse_matrix[j, k] = -1
            sparse_matrix[j, i] = 1

        elif image[h - 1, w]:
            dz[j] = -n_y / n_z
            k = meshgrid[h, w - 1]
            sparse_matrix[j, k] = 1
            sparse_matrix[j, i] = -1

    return sparse_matrix, dz


def get_surface_reconstruction(z, mask, s):
    nonzero_h, nonzero_w = np.where(mask != 0)
    normalized_z = (z - np.mean(z)) / np.std(z)
    outliner_idx = np.abs(normalized_z) > 2
    z_max = np.max(z[~outliner_idx])
    z_min = np.min(z[~outliner_idx])
    Z = mask.astype(np.float32)

    for i in range(s):
        if z[i] > z_max:
            Z[nonzero_h[i], nonzero_w[i]] = z_max
        elif z[i] < z_min:
            Z[nonzero_h[i], nonzero_w[i]] = z_min
        else:
            Z[nonzero_h[i], nonzero_w[i]] = z[i]
    return Z


def get_height(image, surfaceNormals):
    surfaceNormals = np.reshape(surfaceNormals, (image_height, image_width, 3))

    pixels = np.size(np.where(image != 0)[0])

    spare_matrix = scipy.sparse.lil_matrix((2 * pixels, pixels))
    dz = np.zeros((2 * pixels, 1))

    spare_matrix, dz = sparse_multiplcation(spare_matrix, dz, surfaceNormals, image)

    z = scipy.sparse.linalg.spsolve(spare_matrix.T @ spare_matrix, spare_matrix.T @ dz)
    return get_surface_reconstruction(z, image, pixels)


if __name__ == '__main__':
    data_dir = os.path.join('..', 'data')
    out_dir = os.path.join('..', 'output', 'photometricStereo','Extra')
    image_dir = os.path.join(data_dir, 'photometricStereo', 'Extra', 'bunny')

    ImArray, lightDirs = prepare_data(image_dir)
    surfaceNormals = getSurface(ImArray, lightDirs)

    image = read_images(image_dir + '/pic1.bmp')

    Z = get_height(image, surfaceNormals)

    plot_surface_normals(surfaceNormals)
    reflectance_map(Z, out_dir)
    save_point_cloud(Z, out_dir + '/bunny.ply')
    open_point_cloud(out_dir + '/bunny.ply')
    plt.show()

