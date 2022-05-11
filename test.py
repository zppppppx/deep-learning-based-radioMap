import imp
import torch
from modules.Convolutional_RE_Net import map_generate
from utils import Config

opt = Config.Config()
mapseed = opt.shapeUnits.units

Map = map_generate(opt)
out = Map(mapseed)

print(out.shape)