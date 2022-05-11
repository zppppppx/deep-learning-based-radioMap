from PIL.Image import NONE
import torch
import matplotlib.pyplot as plt


class fundamentalShape:
    def __init__(self) -> None:
        square = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0]
        ])
        rectangular = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0]
        ])
        triangle_ortho = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0],
            [0,1,1,0,0,0,0,0,0,0,0],
            [0,1,1,1,0,0,0,0,0,0,0],
            [0,1,1,1,1,0,0,0,0,0,0],
            [0,1,1,1,1,1,0,0,0,0,0],
            [0,1,1,1,1,1,1,0,0,0,0],
            [0,1,1,1,1,1,1,1,0,0,0],
            [0,1,1,1,1,1,1,1,1,0,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0]
        ])
        circle = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,0,0,0,0],
            [0,0,1,1,1,1,1,1,1,0,0],
            [0,0,1,1,1,1,1,1,1,0,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,0,1,1,1,1,1,1,1,0,0],
            [0,0,1,1,1,1,1,1,1,0,0],
            [0,0,0,0,1,1,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0]
        ])
        ellipse = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,1,1,1,1,0,0,0],
            [0,0,1,1,1,1,1,1,1,0,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,0,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0]
        ])
        square_holo = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,1,0,0,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,0,0,1,0],
            [0,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0]
        ])

        self.size = square.shape
        self.shapes = [square, square_holo, rectangular, triangle_ortho, circle, ellipse]
        self.num = len(self.shapes)
        self.units = self._unitsCombine(self.shapes)

    def _unitsCombine(self, shapes):
        out = torch.tensor([])
        for shape in shapes:
            shape_i = shape[None,]
            out = torch.cat([out, shape_i], dim=0)

        return out[None]


class Config:
    map_order = 2

    # PAY ATTENTION: We assign the resolution to 3
    resolution = 3.
    map_size = [94, 94] # 3class for [94, 94] 2class for [101, 101]. Simulation for [114, 114].

    RE_Net = 'Linear' # 'Conv' for Convolutional RE-Net and 'Linear' for Linear RE-Net
    data_fraction = {"3class":0.012, "2class":0.0098} # To assure the amount used is close to 10000

    lr = 1e-5
    lr_param = 1e-2
    lr_map = 1e-4 if RE_Net == 'Conv' else 1e-2

    batchsize = 128
    epoch = 5
    epoch_param = 1
    epoch_map = 2
    epoch_finetune = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    shapeUnits = fundamentalShape()
    metainfo = shapeUnits.units if RE_Net == 'Conv' else torch.ones(1,1, requires_grad=True)

    


if __name__ == '__main__':
    shape = fundamentalShape()
    print(shape.units.shape)
    plt.imshow(shape.shapes[0])
    plt.show()