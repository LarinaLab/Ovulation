import nrrd
import numpy as np
import torch
import torch.nn as nn

from imagejroi import load_imagej_roi_points


g = torch.Generator()
g.manual_seed(423978)

# A particle detector model:
bs = 9
r = bs // 2
model = nn.Sequential(
    nn.Conv3d(1, 1, kernel_size=3),
    nn.Conv3d(1, 1, kernel_size=3),
    nn.SELU(inplace=True),
    nn.Conv3d(1, 1, kernel_size=3),
    nn.Conv3d(1, 1, kernel_size=3),
    nn.Sigmoid()
)

label = "Vivo_Control_hCG14h_040824"
img, nrrd_header = nrrd.read(f"registered/{label}_reg.nrrd", index_order="C")

mean = img.mean().astype(np.float32)
std = img.std().astype(np.float32)
img = torch.tensor((img - mean) / std)

points_plus = load_imagej_roi_points(f"adaptive_filter/{label}_plus.roi")
points_minus = load_imagej_roi_points(f"adaptive_filter/{label}_minus.roi")
points_plus = torch.tensor(points_plus)
points_minus = torch.tensor(points_minus)

points = torch.concat((points_plus, points_minus))
n = points.shape[0]
X = torch.empty((n, 1, bs, bs, bs), dtype=torch.float32)
Y = torch.empty((n, 1), dtype=torch.float32)
keep = torch.empty((n,), dtype=bool)
for i in range(n):
    x, y, t = points[i, :]
    x = round(x.item())
    y = round(y.item())
    t = int(t.item())
    t0 = t - r
    t1 = t + r + 1
    y0 = y - r
    y1 = y + r + 1
    x0 = x - r
    x1 = x + r + 1
    keep[i] = not (t0 < 0 or x0 < 0 or y0 < 0
        or t1 > img.shape[0] or y1 > img.shape[1] or x1 > img.shape[2])
    if keep[i]:
        X[i, 0, :, :, :] = img[t0:t1, y0:y1, x0:x1]
        Y[i, 0] = i < points_plus.shape[0]
X = X[keep, :, :, :, :]
Y = Y[keep, :]

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(20):
    for k in range(100):
        y_out = model(X)
        y_out = y_out.reshape((-1, 1))
        j = loss(y_out, Y)
        accuracy = torch.sum((y_out > 0.5) == Y) / Y.shape[0]
        print(i, k, "acc: ", accuracy.item(), "loss:", j.item())

        optimizer.zero_grad()
        j.backward()
        optimizer.step()
    #input()

torch.save(model.state_dict(), "adaptive_filter/particle_detector.model")

##import time
##img_x = img[None, None, :1004, :, :]
##start = time.time()
##img_y = model(img_x)
##print("took", time.time() - start)
##out = img_y.detach().numpy()
##out = out.squeeze()
##out *= 255
##out = out.astype(np.uint8)
##
##nrrd.write(f"{label}_particles.nrrd", out, header=nrrd_header, index_order="C")
